# based on https://github.com/hongbozheng/transformer/avg_meter.py with extensions
# for tracking learning rate, gradient norms, and embedding quality
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class AverageMeter:
    def __init__(self) -> None:
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricsTracker:
    def __init__(self):
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.learning_rate = []
        self.grad_norms = []
        # per-epoch history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': [],
        }

    def reset_epoch(self):
        self.train_loss.reset()
        self.val_loss.reset()

    def update_train_loss(self, loss: float, batch_size: int = 1):
        self.train_loss.update(loss, n=batch_size)

    def update_val_loss(self, loss: float, batch_size: int = 1):
        self.val_loss.update(loss, n=batch_size)

    def update_lr(self, lr: float):
        self.learning_rate.append(lr)

    def update_grad_norm(self, grad_norm: float):
        self.grad_norms.append(grad_norm)

    def end_epoch(self, epoch: int):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(self.train_loss.avg)
        self.history['val_loss'].append(self.val_loss.avg)
        if self.learning_rate:
            avg_lr = float(np.mean(self.learning_rate[-len(self.learning_rate):]))
            self.history['learning_rate'].append(avg_lr)
        else:
            self.history['learning_rate'].append(0.0)

    def get_current_metrics(self) -> Dict[str, float]:
        return {
            'train_loss': self.train_loss.avg,
            'val_loss': self.val_loss.avg,
            'learning_rate': self.learning_rate[-1] if self.learning_rate else 0.0,
        }

    def get_history(self) -> Dict[str, List[float]]:
        return self.history


def compute_gradient_norm(model: nn.Module) -> float:
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_embedding_quality(query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor,
                              negative_embeddings: torch.Tensor) -> Dict[str, float]:
    """Compute embedding quality metrics
    Args:
        query_embeddings: [B, D] query embeddings
        positive_embeddings: [B, D] positive (equivalent) embeddings
        negative_embeddings: [B, N_neg, D] negative embeddings
    Returns:
        metrics dict with:
            - intra_class_similarity: avg cosine sim between query and positive
            - inter_class_similarity: avg cosine sim between query and negatives
            - separation_margin: intra - inter (higher is better)"""
    # normalize
    query_norm = F.normalize(query_embeddings, p=2, dim=-1)
    pos_norm = F.normalize(positive_embeddings, p=2, dim=-1)
    neg_norm = F.normalize(negative_embeddings, p=2, dim=-1)
    # intra-class similarity (query vs positive)
    intra_sim = torch.sum(query_norm * pos_norm, dim=-1)  # [B]
    intra_class_similarity = intra_sim.mean().item()
    # inter-class similarity (query vs negatives)
    query_expanded = query_norm.unsqueeze(1)  # [B, 1, D]
    inter_sim = torch.sum(query_expanded * neg_norm, dim=-1)  # [B, N_neg]
    inter_class_similarity = inter_sim.mean().item()
    # separation margin
    separation_margin = intra_class_similarity - inter_class_similarity
    return {
        'intra_class_similarity': intra_class_similarity,
        'inter_class_similarity': inter_class_similarity,
        'separation_margin': separation_margin,
    }
