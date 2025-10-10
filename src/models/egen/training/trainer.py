import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm
import logging

from src.models.egen.training.checkpointing import save_best_model, save_last_checkpoint, save_periodic_checkpoint
logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        checkpoint_dir: Path,
        grad_clip_norm: float = 1.0,
        save_every_n_iters: int = 1000,
        log_every_n_iters: int = 100,
    ):
        """
        Args:
            model: encoder model
            criterion: loss function (e.g., InfoNCE)
            optimizer: optimizer
            scheduler: learning rate scheduler (optional)
            train_loader: training data loader
            val_loader: validation data loader (optional)
            device: device to train on
            checkpoint_dir: directory to save checkpoints
            grad_clip_norm: gradient clipping max norm
            save_every_n_iters: save checkpoint every N iterations
            log_every_n_iters: log progress every N iterations
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip_norm = grad_clip_norm
        self.save_every_n_iters = save_every_n_iters
        self.log_every_n_iters = log_every_n_iters
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.global_iteration = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """Args: [epoch: current epoch number], Returns: average training loss for epoch
        Training loop structure from E-Gen reference: https://github.com/hongbozheng/transformer/train.py
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {epoch}",
            leave=True
        )
        for batch_idx, batch in progress_bar:
            query = batch['query'].to(self.device)
            positive = batch['positive'].to(self.device)
            negatives = batch['negatives'].to(self.device)
            query_mask = batch['query_mask'].to(self.device)
            positive_mask = batch['positive_mask'].to(self.device)
            negatives_mask = batch['negatives_mask'].to(self.device)

            query_hidden = self.model(query, mask=query_mask)  # [B, L, D]
            positive_hidden = self.model(positive, mask=positive_mask)  # [B, L, D]
            query_emb = self.model.mean_pool(query_hidden, mask=query_mask)  # [B, D]
            positive_emb = self.model.mean_pool(positive_hidden, mask=positive_mask)  # [B, D]
            batch_size, n_negatives, seq_len = negatives.shape
            negatives_flat = negatives.view(batch_size * n_negatives, seq_len)  # [B*N_neg, L]
            negatives_mask_flat = negatives_mask.view(batch_size * n_negatives, seq_len)  # [B*N_neg, L]
            negatives_hidden = self.model(negatives_flat, mask=negatives_mask_flat)  # [B*N_neg, L, D]
            negatives_emb_flat = self.model.mean_pool(negatives_hidden, mask=negatives_mask_flat)  # [B*N_neg, D]
            embed_dim = negatives_emb_flat.size(-1)
            negatives_emb = negatives_emb_flat.view(batch_size, n_negatives, embed_dim) # [B, N_neg, D]
            loss = self.criterion(query=query_emb, pos_key=positive_emb, neg_key=negatives_emb)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self._get_lr():.2e}'})
            if self.global_iteration % self.log_every_n_iters == 0:
                logger.info(f"Epoch {epoch} | Iter {self.global_iteration} | Loss: {loss.item():.4f} | LR: {self._get_lr():.2e}")
            if self.global_iteration % self.save_every_n_iters == 0 and self.global_iteration > 0:
                save_periodic_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    iteration=self.global_iteration,
                    loss=loss.item(),
                    checkpoint_dir=self.checkpoint_dir,
                    save_every_n_iters=self.save_every_n_iters,
                    best_val_loss=self.best_val_loss
                )
            self.global_iteration += 1
        if self.scheduler is not None:
            self.scheduler.step()
        avg_epoch_loss = total_loss / num_batches
        return avg_epoch_loss

    def validate(self, epoch: int) -> float:
        if self.val_loader is None:
            logger.warning("no validation loader provided, skipping validation")
            return float('inf')
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation {epoch}", leave=False):
                query = batch['query'].to(self.device)
                positive = batch['positive'].to(self.device)
                negatives = batch['negatives'].to(self.device)
                query_mask = batch['query_mask'].to(self.device)
                positive_mask = batch['positive_mask'].to(self.device)
                negatives_mask = batch['negatives_mask'].to(self.device)

                query_hidden = self.model(query, mask=query_mask)
                positive_hidden = self.model(positive, mask=positive_mask)
                query_emb = self.model.mean_pool(query_hidden, mask=query_mask)
                positive_emb = self.model.mean_pool(positive_hidden, mask=positive_mask)
                batch_size, n_negatives, seq_len = negatives.shape
                negatives_flat = negatives.view(batch_size * n_negatives, seq_len)
                negatives_mask_flat = negatives_mask.view(batch_size * n_negatives, seq_len)
                negatives_hidden = self.model(negatives_flat, mask=negatives_mask_flat)
                negatives_emb_flat = self.model.mean_pool(negatives_hidden, mask=negatives_mask_flat)
                embed_dim = negatives_emb_flat.size(-1)
                negatives_emb = negatives_emb_flat.view(batch_size, n_negatives, embed_dim)

                loss = self.criterion(query=query_emb, pos_key=positive_emb, neg_key=negatives_emb)
                total_loss += loss.item()

        avg_val_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self, num_epochs: int, start_epoch: int = 1) -> None:
        logger.info("=" * 60)
        logger.info(f"starting training from epoch {start_epoch} to {num_epochs}")
        logger.info(f"model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"training batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"validation batches: {len(self.val_loader)}")
        logger.info(f"checkpoint directory: {self.checkpoint_dir}")
        logger.info("=" * 60)

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}")
            # validate
            if self.val_loader is not None:
                val_loss = self.validate(epoch)
                self.best_val_loss = save_best_model(
                    model=self.model,
                    checkpoint_dir=self.checkpoint_dir,
                    val_loss=val_loss,
                    best_loss=self.best_val_loss,
                    epoch=epoch
                )
            else:
                val_loss = None
            save_last_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                iteration=self.global_iteration,
                loss=train_loss,
                checkpoint_dir=self.checkpoint_dir,
                best_val_loss=self.best_val_loss
            )
        logger.info("=" * 60)
        logger.info("training complete!")
        logger.info(f"best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"checkpoints saved to: {self.checkpoint_dir}")
        logger.info("=" * 60)

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
