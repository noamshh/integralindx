import logging
import torch.optim as optim
from torch.nn import Module

logger = logging.getLogger(__name__)

def build_optimizer(model: Module, optimizer_cfg: dict) -> optim.Optimizer:
    name = optimizer_cfg['name'].lower()
    if name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=optimizer_cfg['lr'],
            betas=tuple(optimizer_cfg['betas']),
            eps=optimizer_cfg.get('eps', 1e-8),
            weight_decay=optimizer_cfg.get('weight_decay', 0.01)
        )
    elif name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=optimizer_cfg['lr'],
            betas=tuple(optimizer_cfg['betas']),
            eps=optimizer_cfg.get('eps', 1e-8),
            weight_decay=optimizer_cfg.get('weight_decay', 0.0)
        )
    elif name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=optimizer_cfg['lr'],
            momentum=optimizer_cfg.get('momentum', 0.9),
            weight_decay=optimizer_cfg.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(optimizer: optim.Optimizer, scheduler_cfg: dict, num_epochs: int):
    name = scheduler_cfg['name'].lower()
    if name == 'cosine':
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
        min_lr = scheduler_cfg.get('min_lr', 0.0)
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=min_lr
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            logger.info(f"using warmup ({warmup_epochs} epochs) + cosine annealing scheduler")
            return scheduler
        else:
            logger.info("using cosine annealing scheduler (no warmup)")
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=min_lr
            )
    elif name == 'step':
        step_size = scheduler_cfg.get('step_size', 30)
        gamma = scheduler_cfg.get('gamma', 0.1)
        logger.info(f"using step LR scheduler (step_size={step_size}, gamma={gamma})")
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif name == 'none':
        logger.info("no learning rate scheduler")
        return None
    else:
        raise ValueError(f"unknown scheduler: {name}")
