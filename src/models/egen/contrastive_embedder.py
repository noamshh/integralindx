import torch
import numpy as np
import json
import logging
from typing import List, Union, Optional
from pathlib import Path
import sympy as sp
from omegaconf import OmegaConf

from src.models.base_embedder import BaseEmbedder
from src.models.egen.contrastive_model import MathEncoder
from src.models.egen.tokenizer import MathTokenizer
from src.utils.paths import get_paths
from src.utils.prefix_notation import sympy_to_prefix

logger = logging.getLogger(__name__)

paths = get_paths()
PROJECT_ROOT = paths['project_root']

class ContrastiveLearningEmbedder(BaseEmbedder):
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
        self.model = MathEncoder(
            vocab_size=vocab_size,
            dim=encoder_cfg.dim,
            num_layers=encoder_cfg.num_layers,
            num_heads=encoder_cfg.num_heads,
            feedforward_dim=encoder_cfg.feedforward_dim,
            max_seq_len=encoder_cfg.max_seq_len,
            dropout=encoder_cfg.dropout
        ).to(self.device)

        self.tokenizer = MathTokenizer()
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
                expr = sp.sympify(expr_str)
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
    def load(cls, path: Path) -> 'ContrastiveLearningEmbedder':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            config_path = Path(meta['config_path'])
        else:
            config_path = checkpoint.get('config_path', None)
            if config_path:
                config_path = Path(config_path)
        vocab_size = checkpoint['vocab_size']
        embedder = cls(vocab_size=vocab_size, config_path=config_path)
        embedder.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"loaded contrastive embedder from {path}")
        return embedder
