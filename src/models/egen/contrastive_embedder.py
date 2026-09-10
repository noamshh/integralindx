import torch
import numpy as np
import json
import logging
from typing import List, Union, Optional
from pathlib import Path
import sympy as sp
from src.utils.safe_math import CORPUS_MAX_EXPR_LENGTH, safe_sympify
from omegaconf import OmegaConf

from src.models.base_embedder import BaseEmbedder
from src.models.egen.contrastive_model import Encoder
from src.models.egen.tokenizer import Tokenizer
from src.models.egen.vocab import get_vocab_size
from src.utils.paths import get_paths
from src.utils.prefix_notation import sympy_to_prefix

logger = logging.getLogger(__name__)

paths = get_paths()
PROJECT_ROOT = paths['project_root']

class CLEmbedder(BaseEmbedder):
    def __init__(self, vocab_size: int, config_path: Optional[Path] = None, device: Optional[str] = None):
        """Args:
            vocab_size: vocabulary size (from tokenizer)
            config_path: path to model config (default: config/model/ii-cl-19m.yaml)
            device: torch device (default: auto-detect)"""
        if config_path is None:
            config_path = PROJECT_ROOT / 'config' / 'model' / 'ii-cl-19m.yaml'
        config = OmegaConf.load(config_path)
        encoder_cfg = config.encoder
        self.embedding_dim = encoder_cfg.dim
        self.max_seq_len = encoder_cfg.max_seq_len
        self.config_path = config_path
        super().__init__(embedding_dim=self.embedding_dim, method='contrastive')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = Encoder(
            vocab_size=vocab_size,
            dim=encoder_cfg.dim,
            num_layers=encoder_cfg.num_layers,
            num_heads=encoder_cfg.num_heads,
            feedforward_dim=encoder_cfg.feedforward_dim,
            max_seq_len=encoder_cfg.max_seq_len,
            dropout=encoder_cfg.dropout
        ).to(self.device)

        self.tokenizer = Tokenizer()
        logger.info(
            f"initialized contrastive embedder: {self.embedding_dim}D, "
            f"{encoder_cfg.num_layers}L, {encoder_cfg.num_heads}H, "
            f"device={self.device}"
        )

    def encode(self, expressions: Union[str, List[str]]) -> np.ndarray:
        if isinstance(expressions, str):
            expressions = [expressions]
        prefix_exprs = []
        for expr_str in expressions:
            try:
                expr = safe_sympify(expr_str, max_length=CORPUS_MAX_EXPR_LENGTH)
                prefix = sympy_to_prefix(expr)
                prefix_exprs.append(prefix)
            except Exception as e:
                logger.error(f"failed to convert expression '{expr_str}' to prefix: {e}")
                raise ValueError(f"invalid expression: {expr_str}") from e
        token_ids_list = []
        for prefix in prefix_exprs:
            try:
                token_ids = self.tokenizer.encode(prefix, max_len=self.max_seq_len)
                token_ids_list.append(token_ids)
            except ValueError as e:
                logger.error(f"tokenization failed for '{prefix}': {e}")
                raise
        token_ids_tensor = torch.tensor(token_ids_list, dtype=torch.long).to(self.device)
        mask = (token_ids_tensor == self.tokenizer.PAD_ID)
        self.model.eval()
        with torch.no_grad():
            hidden_states = self.model(token_ids_tensor, mask=mask)  # [B, L, D]
            embeddings = self.model.mean_pool(hidden_states, mask=mask)  # [B, D]
        embeddings_np = embeddings.cpu().numpy()
        return embeddings_np

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': len(self.tokenizer),
            'embedding_dim': self.embedding_dim,
            'config_path': str(self.config_path),
        }, path)
        meta_path = path.with_suffix('.meta.json')
        meta = {
            'method': self.method,
            'embedding_dim': self.embedding_dim,
            'vocab_size': len(self.tokenizer),
            'config_path': str(self.config_path),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"saved contrastive embedder to {path}")

    @classmethod
    def load(cls, path: Path, config_path: Optional[Path] = None) -> 'CLEmbedder':
        """load embedder from checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        if config_path is None:
            meta_path = path.with_suffix('.meta.json')
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                config_path = Path(meta['config_path'])
                logger.info(f"config loaded from .meta.json: {config_path}")
            elif 'config_path' in checkpoint:
                config_path = Path(checkpoint['config_path'])
                logger.info(f"config loaded from checkpoint: {config_path}")
            else:
                raise ValueError(
                    f"config_path not found in checkpoint or metadata. "
                    f"Please specify explicitly via config_path parameter.\n"
                    f"Example: embedder.load(checkpoint_path, config_path=Path('config/model/ii-cl-400k.yaml'))"
                )
        else:
            config_path = Path(config_path)
            logger.info(f"using config: {config_path}")

        vocab_size = get_vocab_size()
        embedder = cls(vocab_size=vocab_size, config_path=config_path)
        embedder.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"loaded contrastive embedder from {path}")
        return embedder
