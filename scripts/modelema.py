import math
import torch
from copy import deepcopy

class ModelEMA:
    def __init__(self, model, decay, device=None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema.to(device=device)
        self.updates = 0
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        self.updates += 1
        with torch.no_grad():
            d = self.decay * (1 - math.exp(-self.updates / 2000))
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def get_model(self):
        return self.ema