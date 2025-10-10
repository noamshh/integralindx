import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class TrainingLogger:
    def __init__(
        self,
        log_dir: Path,
        experiment_name: str,
        log_level: str = 'INFO',
        log_to_file: bool = True,
        log_to_console: bool = True,
    ):
        """
        Args:
            log_dir: directory to save logs
            experiment_name: name of experiment
            log_level: logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_file: enable file logging
            log_to_console: enable console logging
        Inspired by https://github.com/hongbozheng/transformer/logger.py but using python logging module
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(f'training.{experiment_name}')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []
        if log_to_file:
            log_file = self.log_dir / 'training.log'
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        self.stats_file = self.log_dir / 'stats.json'
        self.stats_history = []
        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    self.stats_history = json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"could not load existing stats from {self.stats_file}")
        self.logger.info(f"initialized training logger for {experiment_name}")
        self.logger.info(f"log directory: {self.log_dir}")

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  learning_rate: Optional[float] = None, **extra_metrics):
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'timestamp': datetime.now().isoformat(),
        }
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate
        metrics.update(extra_metrics)
        log_msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        if learning_rate is not None:
            log_msg += f" | LR: {learning_rate:.2e}"
        self.logger.info(log_msg)
        self.stats_history.append(metrics)
        self._save_stats()

    def log_iteration(self, epoch: int, iteration: int, loss: float, learning_rate: Optional[float] = None, **extra_metrics):
        log_msg = f"Epoch {epoch} | Iter {iteration:6d} | Loss: {loss:.4f}"
        if learning_rate is not None:
            log_msg += f" | LR: {learning_rate:.2e}"
        self.logger.debug(log_msg)

    def log_checkpoint(self, checkpoint_path: Path, epoch: int, loss: float):
        self.logger.info(f"saved checkpoint: {checkpoint_path.name} (epoch={epoch}, loss={loss:.4f})")

    def log_best_model(self, val_loss: float, epoch: int):
        self.logger.info(f"✓ new best model: val_loss={val_loss:.4f} (epoch={epoch})")

    def log_training_start(self, num_epochs: int, train_batches: int, val_batches: int, model_params: int):
        self.logger.info("=" * 60)
        self.logger.info(f"starting training for {num_epochs} epochs")
        self.logger.info(f"model parameters: {model_params:,}")
        self.logger.info(f"training batches: {train_batches}")
        self.logger.info(f"validation batches: {val_batches}")
        self.logger.info("=" * 60)

    def log_training_end(self, best_val_loss: float):
        self.logger.info("=" * 60)
        self.logger.info("training complete!")
        self.logger.info(f"best validation loss: {best_val_loss:.4f}")
        self.logger.info(f"logs saved to: {self.log_dir}")
        self.logger.info("=" * 60)

    def _save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats_history, f, indent=2)

    def get_stats_history(self) -> list:
        return self.stats_history


def setup_logging(log_dir: Path, experiment_name: str, log_level: str = 'INFO') -> TrainingLogger:
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_level=log_level,
    )
