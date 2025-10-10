import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class TensorBoardLogger:
    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Args:
            log_dir: base directory for logs (e.g., data/models/runs/)
            experiment_name: name of experiment
        """
        self.log_dir = Path(log_dir) / experiment_name / 'tensorboard'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"tensorboard logs: {self.log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_model_gradients(self, model: torch.nn.Module, step: int):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)

    def log_model_weights(self, model: torch.nn.Module, step: int):
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)

    def log_embeddings(self, embeddings: torch.Tensor, labels: Optional[List[str]] = None, tag: str = 'embeddings', step: int = 0):
        if labels is not None:
            metadata = labels
        else:
            metadata = None
        self.writer.add_embedding(embeddings, metadata=metadata, tag=tag, global_step=step)

    def close(self):
        self.writer.close()


def plot_training_curves(stats_file: Path, output_dir: Path, show_lr: bool = True):
    """plot training and validation loss curves from stats.json
    Args:
        stats_file: path to stats.json file
        output_dir: directory to save plots
        show_lr: include learning rate subplot"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_file) as f:
        stats = json.load(f)
    epochs = [s['epoch'] for s in stats]
    train_loss = [s['train_loss'] for s in stats]
    val_loss = [s.get('val_loss', None) for s in stats]
    learning_rate = [s.get('learning_rate', None) for s in stats]
    val_loss_filtered = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]
    if val_loss_filtered:
        val_epochs, val_losses = zip(*val_loss_filtered)
    else:
        val_epochs, val_losses = [], []
    lr_filtered = [(e, lr) for e, lr in zip(epochs, learning_rate) if lr is not None]
    if lr_filtered:
        lr_epochs, lrs = zip(*lr_filtered)
    else:
        lr_epochs, lrs = [], []
    if show_lr and lr_filtered:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
    if val_losses:
        ax1.plot(val_epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if show_lr and lr_filtered:
        ax2.plot(lr_epochs, lrs, label='Learning Rate', color='green', marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"saved training curves to {plot_path}")


def plot_embedding_space(embeddings: np.ndarray, labels: Optional[List[str]] = None, output_path: Optional[Path] = None,
                         method: str = 'tsne', perplexity: int = 30, n_components: int = 2):
    """visualize embedding space
    Args:
        embeddings: [N, D] embedding vectors
        labels: list of N labels
        output_path: path to save plot
        method: 'tsne' or 'umap'
        perplexity: t-SNE perplexity parameter
        n_components: dimensionality (2 or 3)"""
    if method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            logger.error("scikit-learn required for t-SNE, install with: pip install scikit-learn")
            return
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            logger.error("umap-learn required for UMAP, install with: pip install umap-learn")
            return
    else:
        raise ValueError(f"unknown method: {method}, choose 'tsne' or 'umap'")

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is None:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10)
    else:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=label, alpha=0.6, s=10, color=colors[i])
        ax.legend()
    ax.set_title(f'Embedding Space ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"saved embedding plot to {output_path}")
    else:
        plt.show()
    plt.close()
