import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    iteration: int,
    loss: float,
    checkpoint_path: Path,
    best_loss: Optional[float] = None,
    **extra_state
) -> None:
    """
    Save training checkpoint.
    Args:
        model: model to save
        optimizer: optimizer state
        scheduler: learning rate scheduler (optional)
        epoch: current epoch number
        iteration: current iteration number
        loss: current loss value
        checkpoint_path: path to save checkpoint
        best_loss: best validation loss so far (optional)
        **extra_state: additional state to save (e.g., vocab_size, config)
    Checkpoint structure from E-Gen reference: https://github.com/hongbozheng/transformer/train.py
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    checkpoint.update(extra_state)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"saved checkpoint: {checkpoint_path} (epoch={epoch}, iter={iteration}, loss={loss:.4f})")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    Args:
        checkpoint_path: path to checkpoint file
        model: model to load state into
        optimizer: optimizer to load state into (optional)
        scheduler: scheduler to load state into (optional)
        device: device to load tensors to
    Returns:
        checkpoint: checkpoint dict with training state
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"loaded model state from {checkpoint_path}")
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"loaded optimizer state from {checkpoint_path}")
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"loaded scheduler state from {checkpoint_path}")
    epoch = checkpoint.get('epoch', 'unknown')
    iteration = checkpoint.get('iteration', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    logger.info(f"resumed from epoch={epoch}, iter={iteration}, loss={loss}")
    return checkpoint


def save_best_model(
    model: nn.Module,
    checkpoint_dir: Path,
    val_loss: float,
    best_loss: float,
    epoch: int,
    **extra_state
) -> float:
    """
    Save best model if val loss improved
    Args:
        model: model to save
        checkpoint_dir: directory to save checkpoints
        val_loss: current validation loss
        best_loss: best validation loss so far
        epoch: current epoch
        **extra_state: additional state to save
    Returns:
        best_loss: updated best loss
    """
    if val_loss < best_loss:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            **extra_state
        }, best_path)
        logger.info(f"✓ new best model saved: {best_path} (val_loss={val_loss:.4f})")
        return val_loss
    else:
        return best_loss


def save_last_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    iteration: int,
    loss: float,
    checkpoint_dir: Path,
    **extra_state
) -> None:
    """Save last checkpoint (overwriting previous)."""
    last_path = checkpoint_dir / "last_checkpoint.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        iteration=iteration,
        loss=loss,
        checkpoint_path=last_path,
        **extra_state
    )


def save_periodic_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    iteration: int,
    loss: float,
    checkpoint_dir: Path,
    save_every_n_iters: int,
    **extra_state
) -> None:
    if iteration % save_every_n_iters == 0:
        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            iteration=iteration,
            loss=loss,
            checkpoint_path=checkpoint_path,
            **extra_state
        )


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    last_checkpoint = checkpoint_dir / "last_checkpoint.pt"
    if last_checkpoint.exists():
        return last_checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    return checkpoints[-1]
