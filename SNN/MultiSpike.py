import torch
import torch.nn as nn
import torch.nn.functional as F

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None
    
class MultiSpike(nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
    
    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"  

    def forward(self, x): # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)
    
class Multispike(nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
    

    @staticmethod
    def spike_function(x, min_value, max_value, dim=-1):
        x_shape = x.shape
        x_flat = x.reshape(-1, x_shape[dim])
        x_clamped = torch.clamp(x_flat, min=min_value, max=max_value)
        x_rounded = torch.round(x_clamped)
        x_out = x_rounded.reshape(x_shape)
        return x_out
            
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"  

    # def forward(self, x, dim=-1): # B C H W
    #     return self.spike_function(x, min_value=self.min_value, max_value=self.max_value, dim=dim) / (self.Norm)
    
    def forward(self, x, dim=-1): # B C H W
        spikes = self.spike_function(x, min_value=self.min_value, max_value=self.max_value, dim=dim) / self.Norm
        spikes_sum = spikes.sum(dim=dim, keepdim=True)
        spikes_sum = torch.where(spikes_sum > 1e-8, spikes_sum, torch.ones_like(spikes_sum))
        return spikes / spikes_sum
