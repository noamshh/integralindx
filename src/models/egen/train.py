"""
Usage:
    python -m src.models.egen.train --config config/training_config.yaml
    python -m src.models.egen.train --config config/training_config.yaml --resume
    python -m src.models.egen.train --config config/training_config.yaml --experiment my_experiment
"""
import argparse
import logging
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.egen.contrastive_model import MathEncoder
from src.models.egen.datasets.contrastive_dataset import ContrastiveDataset
from src.models.egen.tokenizer import MathTokenizer
from src.models.egen.training.losses import build_criterion
from src.models.egen.training.trainer import ContrastiveTrainer
from src.models.egen.training.checkpointing import load_checkpoint, get_latest_checkpoint
from src.models.egen.training.optimization import build_optimizer, build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_config_path: Path, vocab_size: int, device: torch.device) -> MathEncoder:
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    encoder_cfg = model_cfg['model']['encoder']
    model = MathEncoder(
        vocab_size=vocab_size,
        dim=encoder_cfg['dim'],
        num_layers=encoder_cfg['num_layers'],
        num_heads=encoder_cfg['num_heads'],
        feedforward_dim=encoder_cfg['feedforward_dim'],
        max_seq_len=encoder_cfg['max_seq_len'],
        dropout=encoder_cfg['dropout']
    ).to(device)
    logger.info(f"built model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train E-Gen contrastive embedder")
    parser.add_argument('--config', type=Path, default='config/training_config.yaml', help='path to training config YAML')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--experiment', type=str, help='experiment name (overrides config)')
    parser.add_argument('--device', type=str, help='device to train on (cuda or cpu, overrides config)')
    args = parser.parse_args()
    logger.info(f"loading config from {args.config}")
    config = load_config(args.config)
    if args.device:
        device_name = args.device
    else:
        device_name = config['training']['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    logger.info(f"using device: {device}")
    experiment_name = args.experiment or config['checkpoints']['experiment']
    checkpoint_dir = Path(config['checkpoints']['dir']) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"experiment: {experiment_name}")
    logger.info(f"checkpoint directory: {checkpoint_dir}")
    logger.info("initializing tokenizer...")
    tokenizer = MathTokenizer()
    vocab_size = len(tokenizer.vocab)
    logger.info(f"vocabulary size: {vocab_size}")
    logger.info("loading datasets...")
    dataset_cfg = config['dataset']
    train_dataset = ContrastiveDataset(
        tsv_path=dataset_cfg['train_tsv'],
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg['max_seq_len']
    )
    logger.info(f"training examples: {len(train_dataset)}")
    val_dataset = ContrastiveDataset(
        tsv_path=dataset_cfg['val_tsv'],
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg['max_seq_len']
    )
    logger.info(f"validation examples: {len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=val_dataset.collate_fn
    )
    logger.info("building model...")
    model = build_model(
        model_config_path=Path(config['model']['config_path']),
        vocab_size=vocab_size,
        device=device
    )
    logger.info("building criterion...")
    criterion = build_criterion(**config['criterion'])
    logger.info("building optimizer...")
    optimizer = build_optimizer(model, config['optimizer'])
    logger.info("building scheduler...")
    scheduler = build_scheduler(
        optimizer,
        config['scheduler'],
        num_epochs=config['training']['n_epochs']
    )
    start_epoch = 1
    if args.resume:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"resuming from checkpoint: {latest_checkpoint}")
            checkpoint = load_checkpoint(
                checkpoint_path=latest_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"resuming from epoch {start_epoch}")
        else:
            logger.warning("no checkpoint found, starting from scratch")
    logger.info("initializing trainer...")
    trainer = ContrastiveTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip_norm=config['training']['grad_clip_norm'],
        save_every_n_iters=config['training']['save_every_n_iters'],
        log_every_n_iters=config['training']['log_every_n_iters']
    )
    logger.info("=" * 60)
    logger.info("starting training")
    logger.info("=" * 60)
    trainer.train(
        num_epochs=config['training']['n_epochs'],
        start_epoch=start_epoch
    )
    logger.info("training complete!")


if __name__ == "__main__":
    main()
