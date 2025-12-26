from torch.optim import AdamW
from .deep import DeepMomentumGradientDesent
from torch.nn import Module
from typing import Any, Callable


class OptimizerBuilder(Callable):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def __call__(self, model: Module, **kwargs) -> Any:
        raise NotImplementedError
    
    def clone(self) -> 'OptimizerBuilder':
        raise NotImplementedError


class DeepMomentumGradientDesentBuilder(OptimizerBuilder):
    def __call__(self, model: Module, **kwargs) -> DeepMomentumGradientDesent:
        return DeepMomentumGradientDesent(params=model.named_parameters(), **self.kwargs, **kwargs)
    
    def clone(self) -> 'DeepMomentumGradientDesentBuilder':
        return DeepMomentumGradientDesentBuilder(**self.kwargs)


class AdamWBuilder(OptimizerBuilder):
    def __call__(self, model: Module, **kwargs) -> AdamW:
        optimizer = AdamW(params=model.parameters(), **self.kwargs, **kwargs)
        optimizer.zero_grad()
        return optimizer
    
    def clone(self) -> 'AdamWBuilder':
        return AdamWBuilder(**self.kwargs)
