import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from scripts.config import cfg

SMI_TOKEN_RE = re.compile(
    r"(\%\d{2}|Br|Cl|Si|Se|\[[^\]]+\]|@@|@|=|#|/|\\|\(|\)|\.|:|,|\+|\-|\*|\$|\^|\?|>|<|\d+|[A-Za-z])"
)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._precompute_freqs_cis(max_len)

    def _precompute_freqs_cis(self, max_len):
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        self.register_buffer('_cos_cached', cos, persistent=False)
        self.register_buffer('_sin_cached', sin, persistent=False)

    def forward(self, seq_len, device=None, dtype=None):
        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]

        if device is not None and cos.device != device:
            cos = cos.to(device=device)
            sin = sin.to(device=device)

        if dtype is not None and cos.dtype != dtype:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout=0.1, max_len=2048):
        super().__init__()
        assert emb_dim % n_heads == 0

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len=max_len)

        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        cos, sin = self.rope(L, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_flash and mask is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                mask = mask.bool() if mask.dtype != torch.bool else mask
                attn_mask = torch.zeros(B, 1, 1, L, dtype=attn_scores.dtype, device=attn_scores.device)
                attn_mask.masked_fill_(~mask[:, None, None, :], float('-inf'))
                attn_scores = attn_scores + attn_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.proj(attn_output)
        return output

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w3 = nn.Linear(dim_hidden, dim_in, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlockImproved(nn.Module):
    def __init__(self, emb_dim, n_heads, ff_dim, dropout=0.1, drop_path=0.0, max_len=2048):
        super().__init__()

        norm_layer = RMSNorm

        self.norm1 = norm_layer(emb_dim)
        self.attn = MultiHeadAttentionWithRoPE(emb_dim, n_heads, dropout, max_len)
        self.norm2 = norm_layer(emb_dim)
        self.ff = SwiGLU(emb_dim, ff_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class AttentionPooling(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attn_score = nn.Linear(dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.attn_proj(x)

        scores = self.attn_score(h).squeeze(-1)

        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled

def _smiles_tokenize(smi: str):
    if smi is None:
        return []
    toks = SMI_TOKEN_RE.findall(smi)
    if not toks:
        return list(smi)
    return toks

class SmilesTokenizer:
    def __init__(self, vocab_tokens=None, unk_token='[UNK]',
                 pad_token='[PAD]', mask_token='[MASK]'):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        base_tokens = [pad_token, unk_token, mask_token]
        self.tokens = base_tokens + (sorted(vocab_tokens) if vocab_tokens else [])
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab_size = len(self.tokens)

    def build_vocab(self, smiles_list):
        tok_set = set()
        for smi in smiles_list:
            if not isinstance(smi, str):
                continue
            toks = _smiles_tokenize(smi.strip())
            tok_set.update(toks)
        sorted_tokens = sorted(tok_set)
        base_tokens = [self.pad_token, self.unk_token, self.mask_token]
        self.tokens = base_tokens + sorted_tokens
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab_size = len(self.tokens)

    def encode(self, smiles, max_length=None):
        toks = _smiles_tokenize(smiles) if isinstance(smiles, str) else []
        ids = [self.token_to_id.get(t, self.token_to_id[self.unk_token]) for t in toks]
        if max_length is not None:
            if len(ids) < max_length:
                ids += [self.token_to_id[self.pad_token]] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
        return ids

    def decode(self, ids):
        toks = [self.id_to_token.get(i, self.unk_token) for i in ids]
        return ''.join([t for t in toks if t not in (self.pad_token,)])

class CHRTModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim=cfg.emb_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            drop_path_rate=cfg.drop_path_rate,
            cls_dropout=cfg.cls.dropout,
            num_classes=None,
            tie_word_embeddings=True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.tie_word_embeddings = tie_word_embeddings

        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.blocks = nn.ModuleList([
            TransformerBlockImproved(
                emb_dim=emb_dim,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                drop_path=dpr[i],
                max_len=max_len
            ) for i in range(n_layers)
        ])

        self.norm = RMSNorm(emb_dim)
        self.mlm_head = nn.Linear(emb_dim, vocab_size)

        self.attn_pool = AttentionPooling(emb_dim, dropout=cls_dropout)
        self.cls_dropout = nn.Dropout(cls_dropout)

        self.classifier = None
        if num_classes is not None:
            self.classifier = self.get_classifier_head(num_classes)

        self.apply(self._init_weights)

        if tie_word_embeddings:
            self._tie_weights()

    def _tie_weights(self):
        self.mlm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, return_hidden=False):
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        mask = attention_mask.bool() if attention_mask is not None else None

        checkpoint_layers = len(self.blocks) // 2

        for i, block in enumerate(self.blocks):
            if i < checkpoint_layers:
                x = checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        x = self.norm(x)

        if return_hidden:
            return x, None

        logits = self.mlm_head(x)
        return logits

    def encode(self, input_ids, attention_mask=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                hidden, _ = self.forward(input_ids, attention_mask, return_hidden=True)
        else:
            hidden, _ = self.forward(input_ids, attention_mask, return_hidden=True)
        return hidden

    def sequence_repr(self, input_ids, attention_mask):
        hidden, _ = self.forward(input_ids, attention_mask, return_hidden=True)
        pooled = self.attn_pool(hidden, attention_mask)
        return self.cls_dropout(pooled)

    def get_classifier_head(self, num_classes, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = self.emb_dim

        return nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            RMSNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )