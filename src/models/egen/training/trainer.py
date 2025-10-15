import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

from src.models.egen.training.checkpointing import save_best_model, save_last_checkpoint, save_periodic_checkpoint
from src.models.egen.training.logger import TrainingLogger
from src.models.egen.training.metrics import MetricsTracker, compute_gradient_norm, compute_embedding_quality
from src.models.egen.training.visualization import TensorBoardLogger, plot_training_curves


class CLTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        logger: TrainingLogger,
        device: torch.device,
        checkpoint_dir: Path,
        grad_clip_norm: float = 1.0,
        save_every_n_iters: int = 1000,
        log_every_n_iters: int = 100,
        validate_every_n_iters: int = 0,
        use_tensorboard: bool = True,
        tensorboard_log_dir: Optional[Path] = None,
        experiment_name: str = "egen_training",
        compute_emb_quality: bool = False,
        emb_quality_every_n_iters: int = 1000,
        vocab_size: Optional[int] = None,
        config_path: Optional[Path] = None,
        evaluation_manager: Optional[Any] = None,
    ):
        """
        Args:
            model: encoder model
            criterion: loss function
            optimizer: optimizer
            scheduler: learning rate scheduler (optional)
            train_loader: training data loader
            val_loader: validation data loader (optional)
            logger: TrainingLogger instance
            device: device to train on
            checkpoint_dir: directory to save checkpoints
            grad_clip_norm: gradient clipping max norm
            save_every_n_iters: save checkpoint every N iterations
            log_every_n_iters: log progress every N iterations
            validate_every_n_iters: run validation every N iterations (0 = only at end of epoch)
            use_tensorboard: enable TensorBoard logging
            tensorboard_log_dir: TensorBoard log directory
            experiment_name: experiment name for TensorBoard
            compute_emb_quality: compute embedding quality metrics during training
            emb_quality_every_n_iters: compute embedding quality every N iterations
            vocab_size: vocabulary size (saved in checkpoints)
            config_path: model config path (saved in checkpoints)
            evaluation_manager: EvaluationManager instance for training-time evaluation (optional)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_clip_norm = grad_clip_norm
        self.save_every_n_iters = save_every_n_iters
        self.log_every_n_iters = log_every_n_iters
        self.validate_every_n_iters = validate_every_n_iters
        self.compute_emb_quality = compute_emb_quality
        self.emb_quality_every_n_iters = emb_quality_every_n_iters
        self.vocab_size = vocab_size
        self.config_path = config_path
        self.evaluation_manager = evaluation_manager
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.global_iteration = 0
        self.best_val_loss = float('inf')
        self.metrics = MetricsTracker()
        self.tb_logger = None
        if use_tensorboard:
            if tensorboard_log_dir is None:
                tensorboard_log_dir = self.checkpoint_dir.parent / 'runs'
            self.tb_logger = TensorBoardLogger(log_dir=tensorboard_log_dir, experiment_name=experiment_name)
            self.logger.info(f"tensorboard logging enabled: {self.tb_logger.log_dir}")

    def train_epoch(self, epoch: int) -> float:
        """Args: [epoch: current epoch number], Returns: average training loss for epoch
        Core loop structure from E-Gen reference: https://github.com/hongbozheng/transformer/train.py
        """
        self.model.train()
        self.metrics.reset_epoch()
        num_batches = len(self.train_loader)
        progress_bar = tqdm(enumerate(self.train_loader), total=num_batches, desc=f"Epoch {epoch}", leave=True)
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

            # compute and track gradient norm
            grad_norm = compute_gradient_norm(self.model)
            self.metrics.update_grad_norm(grad_norm)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            # update metrics
            self.metrics.update_train_loss(loss.item(), batch_size=batch_size)
            self.metrics.update_lr(self._get_lr())

            # update progress bar
            progress_bar.set_postfix({
                'loss': f'{self.metrics.train_loss.avg:.4f}',
                'lr': f'{self._get_lr():.2e}',
                'grad_norm': f'{grad_norm:.3f}'
            })

            # logging
            if self.global_iteration % self.log_every_n_iters == 0:
                self.logger.log_iteration(epoch, self.global_iteration, loss.item(), learning_rate=self._get_lr())

                if self.tb_logger is not None:
                    self.tb_logger.log_scalar('train/loss', loss.item(), self.global_iteration)
                    self.tb_logger.log_scalar('train/learning_rate', self._get_lr(), self.global_iteration)
                    self.tb_logger.log_scalar('train/grad_norm', grad_norm, self.global_iteration)

                if self.compute_emb_quality and self.global_iteration % self.emb_quality_every_n_iters == 0:
                    with torch.no_grad():
                        emb_quality = compute_embedding_quality(query_emb, positive_emb, negatives_emb)
                        self.logger.info(
                            f"embedding quality [iter {self.global_iteration}]: "
                            f"intra={emb_quality['intra_class_similarity']:.4f}, "
                            f"inter={emb_quality['inter_class_similarity']:.4f}, "
                            f"margin={emb_quality['separation_margin']:.4f}"
                        )
                        if self.tb_logger is not None:
                            self.tb_logger.log_scalars('embedding_quality', emb_quality, self.global_iteration)

            # checkpointing
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
                    best_val_loss=self.best_val_loss,
                    vocab_size=self.vocab_size,
                    config_path=str(self.config_path) if self.config_path else None
                )
            self.global_iteration += 1

            # validation
            if self.validate_every_n_iters > 0 and self.global_iteration % self.validate_every_n_iters == 0:
                if self.val_loader is not None:
                    val_loss = self.validate(epoch, iteration=self.global_iteration)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_best_model(
                            model=self.model,
                            checkpoint_dir=self.checkpoint_dir,
                            val_loss=val_loss,
                            best_loss=self.best_val_loss,
                            epoch=epoch,
                            vocab_size=self.vocab_size,
                            config_path=str(self.config_path) if self.config_path else None
                        )
                        self.logger.info(f"new best validation loss: {val_loss:.4f} at iteration {self.global_iteration}")
                    # back to train mode
                    self.model.train()

            # training-time evaluation
            if self.evaluation_manager is not None:
                self.evaluation_manager.run_evaluation_hooks(
                    model=self.model,
                    device=self.device,
                    iteration=self.global_iteration,
                    tb_logger=self.tb_logger,
                    console_logger=self.logger
                )

        if self.scheduler is not None:
            self.scheduler.step()

        avg_epoch_loss = self.metrics.train_loss.avg
        return avg_epoch_loss

    def validate(self, epoch: int, iteration: Optional[int] = None) -> float:
        if self.val_loader is None:
            self.logger.warning("no validation loader provided, skipping validation")
            return float('inf')
        self.model.eval()
        num_batches = len(self.val_loader)
        desc = f"Validation {epoch}" if iteration is None else f"Val (iter {iteration})"
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=desc, leave=False):
                query = batch['query'].to(self.device)
                positive = batch['positive'].to(self.device)
                negatives = batch['negatives'].to(self.device)
                query_mask = batch['query_mask'].to(self.device)
                positive_mask = batch['positive_mask'].to(self.device)
                negatives_mask = batch['negatives_mask'].to(self.device)
                batch_size = query.size(0)

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
                self.metrics.update_val_loss(loss.item(), batch_size=batch_size)

        avg_val_loss = self.metrics.val_loss.avg
        if self.tb_logger is not None:
            x_axis = iteration if iteration is not None else self.global_iteration
            self.tb_logger.log_scalar('val/loss', avg_val_loss, x_axis)

        return avg_val_loss

    def train(self, num_epochs: int, start_epoch: int = 1) -> None:
        model_params = sum(p.numel() for p in self.model.parameters())
        train_batches, val_batches = len(self.train_loader), len(self.val_loader)
        self.logger.log_training_start(num_epochs, train_batches, val_batches, model_params,
                                       self.checkpoint_dir, start_epoch)
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch(epoch)

            # validate
            if self.val_loader is not None:
                val_loss = self.validate(epoch)
                self.best_val_loss = save_best_model(
                    model=self.model,
                    checkpoint_dir=self.checkpoint_dir,
                    val_loss=val_loss,
                    best_loss=self.best_val_loss,
                    epoch=epoch,
                    vocab_size=self.vocab_size,
                    config_path=str(self.config_path) if self.config_path else None
                )
            else:
                val_loss = None

            self.metrics.end_epoch(epoch)

            self.logger.log_epoch(epoch, train_loss, val_loss=val_loss, learning_rate=self._get_lr())
            save_last_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                iteration=self.global_iteration,
                loss=train_loss,
                checkpoint_dir=self.checkpoint_dir,
                best_val_loss=self.best_val_loss,
                vocab_size=self.vocab_size,
                config_path=str(self.config_path) if self.config_path else None
            )

        self.logger.log_training_end(self.best_val_loss, self.checkpoint_dir)
        self._generate_training_curves()
        if self.tb_logger is not None:
            self.tb_logger.close()
            self.logger.info("tensorboard logger closed")

    def _generate_training_curves(self) -> None:
        try:
            stats_file = self.checkpoint_dir / 'stats.json'
            if stats_file.exists():
                self.logger.info("generating training curves from stats.json...")
                plot_training_curves(
                    stats_file=stats_file,
                    output_dir=self.checkpoint_dir,
                    show_lr=True
                )
            else:
                self.logger.warning(f"stats.json not found at {stats_file}, skipping curve generation")
        except Exception as e:
            self.logger.warning(f"failed to generate training curves: {e}")

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
