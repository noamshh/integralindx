import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.models.egen.contrastive_model import Encoder
from src.models.egen.datasets.contrastive_dataset import ContrastiveDataset
from src.models.egen.tokenizer import Tokenizer
from src.models.egen.training.losses import build_criterion
from src.models.egen.training.trainer import CLTrainer
from src.models.egen.training.checkpointing import resume_training_from_checkpoint
from src.models.egen.training.optimization import build_optimizer, build_scheduler
from src.models.egen.training.logger import TrainingLogger, setup_logging
from src.models.egen.training.evaluation import EvaluationManager
from src.utils.paths import get_paths


def build_model(model_cfg: DictConfig, vocab_size: int, device: torch.device, logger: TrainingLogger) -> Encoder:
    encoder_cfg = model_cfg.encoder
    model = Encoder(
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
    # logging
    paths = get_paths()
    log_dir = paths['models']['runs']
    experiment_name = cfg.training.checkpoints.experiment
    train_logger = setup_logging(log_dir=log_dir, experiment_name=experiment_name)

    train_logger.info("configuration:\n" + OmegaConf.to_yaml(cfg))

    device_name = cfg.training.training.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        train_logger.warning("CUDA not available, using CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    train_logger.info(f"using device: {device}")

    # checkpoint directory
    checkpoint_dir = Path(cfg.training.checkpoints.dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_logger.info(f"experiment: {experiment_name}")
    train_logger.info(f"checkpoint directory: {checkpoint_dir}")

    # tokenizer
    train_logger.info("initializing tokenizer...")
    tokenizer = Tokenizer()
    vocab_size = len(tokenizer.vocab)
    train_logger.info(f"vocabulary size: {vocab_size}")

    # datasets
    train_logger.info("loading datasets...")
    dataset_cfg = cfg.training.dataset
    in_memory = dataset_cfg.get('in_memory', False)
    train_logger.info(f"dataset loading mode: 'in-memory'={in_memory}")

    train_dataset = ContrastiveDataset(
        tsv_path=dataset_cfg.train_tsv,
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg.max_seq_len,
        in_memory=in_memory
    )
    train_logger.info(f"training examples: {len(train_dataset)}")

    val_dataset = ContrastiveDataset(
        tsv_path=dataset_cfg.val_tsv,
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg.max_seq_len,
        in_memory=in_memory
    )
    train_logger.info(f"validation examples: {len(val_dataset)}")

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

    train_logger.info("building model...")
    model = build_model(cfg.model, vocab_size, device, train_logger)

    train_logger.info("building criterion...")
    criterion = build_criterion(**OmegaConf.to_container(cfg.training.criterion))

    train_logger.info("building optimizer...")
    optimizer = build_optimizer(model, OmegaConf.to_container(cfg.training.optimizer))

    train_logger.info("building scheduler...")
    scheduler = build_scheduler(
        optimizer,
        OmegaConf.to_container(cfg.training.scheduler),
        num_epochs=cfg.training.training.n_epochs
    )

    # resume from checkpoint
    resume_state = resume_training_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_logger=train_logger
    )
    start_epoch = resume_state['start_epoch']
    checkpoint_loaded = resume_state['checkpoint_loaded']
    resume_best_val_loss = resume_state['best_val_loss']

    # evaluation manager
    evaluation_manager = None
    if hasattr(cfg.training, 'evaluation') and cfg.training.evaluation.get('enabled', False):
        train_logger.info("initializing evaluation manager...")
        evaluation_manager = EvaluationManager.from_config(
            eval_cfg=OmegaConf.to_container(cfg.training.evaluation),
            tokenizer=tokenizer,
            embedding_dim=cfg.model.encoder.dim,
            checkpoint_dir=checkpoint_dir,
        )
        train_logger.info("evaluation manager initialized successfully")

    # trainer
    train_logger.info("initializing trainer...")
    viz_cfg = cfg.training.visualization
    config_path = paths['project_root'] / 'config' / 'model' / f"{cfg.model.name}.yaml"
    trainer = CLTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=train_logger,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip_norm=cfg.training.training.grad_clip_norm,
        save_every_n_iters=cfg.training.training.save_every_n_iters,
        log_every_n_iters=cfg.training.training.log_every_n_iters,
        validate_every_n_iters=cfg.training.training.get('validate_every_n_iters', 0),
        use_tensorboard=viz_cfg.use_tensorboard,
        tensorboard_log_dir=Path(viz_cfg.tensorboard_log_dir) if viz_cfg.tensorboard_log_dir else None,
        experiment_name=experiment_name,
        compute_emb_quality=viz_cfg.compute_embedding_quality,
        emb_quality_every_n_iters=viz_cfg.embedding_quality_every_n_iters,
        vocab_size=vocab_size,
        config_path=config_path,
        evaluation_manager=evaluation_manager,
    )

    # restore state from checkpoint
    if checkpoint_loaded:
        trainer.best_val_loss = resume_best_val_loss
        train_logger.info(f"restored trainer state: best_val_loss={resume_best_val_loss}")

    # train
    trainer.train(
        num_epochs=cfg.training.training.n_epochs,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
