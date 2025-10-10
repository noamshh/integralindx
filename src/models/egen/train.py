import logging
import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.models.egen.contrastive_model import MathEncoder
from src.models.egen.datasets.contrastive_dataset import ContrastiveDataset
from src.models.egen.tokenizer import MathTokenizer
from src.models.egen.training.losses import build_criterion
from src.models.egen.training.trainer import ContrastiveTrainer
from src.models.egen.training.checkpointing import load_checkpoint, get_latest_checkpoint
from src.models.egen.training.optimization import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


def build_model(model_cfg: DictConfig, vocab_size: int, device: torch.device) -> MathEncoder:
    encoder_cfg = model_cfg.encoder
    model = MathEncoder(
        vocab_size=vocab_size,
        dim=encoder_cfg.dim,
        num_layers=encoder_cfg.num_layers,
        num_heads=encoder_cfg.num_heads,
        feedforward_dim=encoder_cfg.feedforward_dim,
        max_seq_len=encoder_cfg.max_seq_len,
        dropout=encoder_cfg.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"built model {model_cfg.name} with {n_params:,} parameters")
    return model


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("configuration:\n" + OmegaConf.to_yaml(cfg))
    device_name = cfg.training.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    logger.info(f"using device: {device}")

    # checkpoint directory
    experiment_name = cfg.training.checkpoints.experiment
    checkpoint_dir = Path(cfg.training.checkpoints.dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"experiment: {experiment_name}")
    logger.info(f"checkpoint directory: {checkpoint_dir}")

    # tokenizer
    logger.info("initializing tokenizer...")
    tokenizer = MathTokenizer()
    vocab_size = len(tokenizer.vocab)
    logger.info(f"vocabulary size: {vocab_size}")

    # datasets
    logger.info("loading datasets...")
    dataset_cfg = cfg.training.dataset
    train_dataset = ContrastiveDataset(
        tsv_path=dataset_cfg.train_tsv,
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg.max_seq_len
    )
    logger.info(f"training examples: {len(train_dataset)}")

    val_dataset = ContrastiveDataset(
        tsv_path=dataset_cfg.val_tsv,
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg.max_seq_len
    )
    logger.info(f"validation examples: {len(val_dataset)}")

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.training.num_workers,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.training.num_workers,
        collate_fn=val_dataset.collate_fn
    )

    # build model
    logger.info("building model...")
    model = build_model(cfg.model, vocab_size, device)

    # build criterion
    logger.info("building criterion...")
    criterion = build_criterion(**OmegaConf.to_container(cfg.training.criterion))

    # build optimizer
    logger.info("building optimizer...")
    optimizer = build_optimizer(model, OmegaConf.to_container(cfg.training.optimizer))

    # build scheduler
    logger.info("building scheduler...")
    scheduler = build_scheduler(
        optimizer,
        OmegaConf.to_container(cfg.training.scheduler),
        num_epochs=cfg.training.training.n_epochs
    )

    # check for resume flag in command line
    start_epoch = 1
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint and latest_checkpoint.exists():
        logger.info(f"found checkpoint: {latest_checkpoint}")
        logger.info("to resume, loading checkpoint...")
        checkpoint = load_checkpoint(
            checkpoint_path=latest_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"resuming from epoch {start_epoch}")

    # trainer
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
        grad_clip_norm=cfg.training.training.grad_clip_norm,
        save_every_n_iters=cfg.training.training.save_every_n_iters,
        log_every_n_iters=cfg.training.training.log_every_n_iters
    )
    # train
    logger.info("=" * 60)
    logger.info("starting training")
    logger.info("=" * 60)
    trainer.train(
        num_epochs=cfg.training.training.n_epochs,
        start_epoch=start_epoch
    )
    logger.info("training complete!")


if __name__ == "__main__":
    main()
