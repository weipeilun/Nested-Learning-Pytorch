from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from nested_learning_pytorch.training import build_model_from_cfg, unwrap_config


@hydra.main(config_path="configs", config_name="hope_tiny", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    device = cfg.train.device
    
    model = build_model_from_cfg(cfg.model).to(device)
    model.is_training = True
    steps = 10
    x = torch.randn(1, 10240, 64)
    y = torch.randn(1, 10240, 64)
    state = None
    x = x.to(device)
    y = y.to(device)
    for step in range(steps):
        print(f"step {step}")
        grads_dict, state = model.forward_inner_loop(x=x, y=y, state=state)
        
        model.outer_update(grads_dict=grads_dict)
        
        # Explicitly clean up to prevent memory accumulation
        grads_dict = None
        state = None
 

if __name__ == "__main__":
    main()
