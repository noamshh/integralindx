import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import faiss
import sympy as sp

from src.database.integral_db import IntegralDatabase
from corpus.formula_models import IntegrandGroup
from src.utils.prefix_notation import sympy_to_prefix
from src.models.egen.tokenizer import Tokenizer
from src.models.egen.training.visualization import plot_embedding_space

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    tier: int
    iteration: int
    timestamp: float

    recall_at_1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: Optional[float] = None  # mean reciprocal rank
    map: Optional[float] = None  # mean average precision

    intra_family_sim: Optional[float] = None
    inter_family_sim: Optional[float] = None
    separation_margin: Optional[float] = None
    family_purity_at_1: Optional[float] = None
    family_purity_at_5: Optional[float] = None
    family_purity_at_10: Optional[float] = None

    n_queries: Optional[int] = None
    n_groups: Optional[int] = None
    eval_time_sec: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class EvaluationManager:
    def __init__(
        self,
        topk_queries_path: Path,
        db_path: Path,
        tokenizer: Tokenizer,
        embedding_dim: int,
        warmup_iterations: int = 10000,
        tier1_freq: int = 5000,
        tier2_freq: int = 25000,
        tier3_freq: int = 50000,
        n_samples_tier2: int = 2000,
        n_viz_samples: int = 1000,
        seed: int = 42,
        checkpoint_dir: Optional[Path] = None,
        analogy_queries_path: Optional[Path] = None,
    ):
        self.topk_queries_path = Path(topk_queries_path)
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.warmup_iterations = warmup_iterations
        self.tier1_freq = tier1_freq
        self.tier2_freq = tier2_freq
        self.tier3_freq = tier3_freq
        self.n_samples_tier2 = n_samples_tier2
        self.n_viz_samples = n_viz_samples
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.analogy_queries_path = Path(analogy_queries_path) if analogy_queries_path else None

        if not self.topk_queries_path.exists():
            raise FileNotFoundError(f"eval queries not found: {self.topk_queries_path}")
        with open(self.topk_queries_path) as f:
            data = json.load(f)
            self.eval_queries = data['queries']
        logger.info(f"loaded {len(self.eval_queries)} evaluation queries")

        if self.analogy_queries_path and self.analogy_queries_path.exists():
            with open(self.analogy_queries_path) as f:
                data = json.load(f)
                self.analogy_queries = data['analogies']
            logger.info(f"loaded {len(self.analogy_queries)} analogy queries")
        else:
            self.analogy_queries = []
            if self.analogy_queries_path:
                logger.warning(f"analogy queries not found: {self.analogy_queries_path}")

        self.db = IntegralDatabase(db_path=self.db_path, fallback_to_jsonl=False)
        logger.info(f"connected to database: {self.db_path}")

        self._all_groups_cache: Optional[List[IntegrandGroup]] = None
        self._tier2_sample_cache: Optional[List[IntegrandGroup]] = None

        logger.info(
            f"initialized evaluation manager: warmup={warmup_iterations}, "
            f"tier1_freq={tier1_freq}, tier2_freq={tier2_freq}, tier3_freq={tier3_freq}"
        )

    @classmethod
    def from_config(cls, eval_cfg: Dict, tokenizer: Any, embedding_dim: int, checkpoint_dir: Path) -> 'EvaluationManager':
        return cls(
            db_path=Path(eval_cfg['db_path']),
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            warmup_iterations=eval_cfg['warmup_iterations'],
            tier1_freq=eval_cfg['tier1_every_n_iters'],
            tier2_freq=eval_cfg['tier2_every_n_iters'],
            tier3_freq=eval_cfg['tier3_every_n_iters'],
            n_samples_tier2=eval_cfg['n_samples_tier2'],
            n_viz_samples=eval_cfg['n_viz_samples'],
            seed=eval_cfg['seed'],
            checkpoint_dir=checkpoint_dir,
            topk_queries_path=Path(eval_cfg['topk_queries_path']),
            analogy_queries_path=Path(eval_cfg['analogy_queries_path']),
        )

    def should_evaluate(self, iteration: int, tier: int) -> bool:
        if iteration < self.warmup_iterations:
            return False
        if tier == 1:
            return iteration % self.tier1_freq == 0
        elif tier == 2:
            return iteration % self.tier2_freq == 0
        elif tier == 3:
            return iteration % self.tier3_freq == 0
        return False

    def run_evaluation(self, model: nn.Module, device: torch.device, iteration: int, tb_logger: Optional[Any] = None,
                       console_logger: Optional[logging.Logger] = None) -> None:
        if iteration < self.warmup_iterations:
            return

        # tier 1
        if self.should_evaluate(iteration, tier=1):
            try:
                metrics = self.evaluate_tier1(model, device)
                if tb_logger is not None:
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            tb_logger.log_scalar(f'eval/tier1/{metric_name}', metric_value, iteration)
            except Exception as e:
                logger.warning(f"Tier 1 evaluation failed at iteration {iteration}: {e}")

        # tier 2
        if self.should_evaluate(iteration, tier=2):
            try:
                metrics = self.evaluate_tier2(model, device)
                if tb_logger is not None:
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            tb_logger.log_scalar(f'eval/tier2/{metric_name}', metric_value, iteration)
            except Exception as e:
                logger.warning(f"Tier 2 evaluation failed at iteration {iteration}: {e}")

        # tier 3
        if self.should_evaluate(iteration, tier=3):
            try:
                results = self.evaluate_tier3(model, device, iteration)
                if tb_logger is not None:
                    for metric_name, metric_value in results['metrics'].items():
                        if isinstance(metric_value, (int, float)):
                            tb_logger.log_scalar(f'eval/tier3/{metric_name}', metric_value, iteration)
                if console_logger and 'search_examples' in results:
                    console_logger.info(results['search_examples'])
            except Exception as e:
                logger.warning(f"Tier 3 evaluation failed at iteration {iteration}: {e}")

    def evaluate_tier1(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        start_time = time.time()
        logger.info("[Tier 1] starting fast retrieval evaluation...")

        # sample 1k groups if not cached
        if self._tier2_sample_cache is None or len(self._tier2_sample_cache) < 1000:
            sample_groups = self.db.sample_groups(n=1000, seed=self.seed, exclude_curated=True)
        else:
            sample_groups = self._tier2_sample_cache[:1000]

        # embed groups
        group_embeddings, group_families = self._embed_groups(model, device, sample_groups)

        # build FAISS index
        index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(group_embeddings)
        index.add(group_embeddings)

        # evaluate queries
        retrieval_metrics = self._evaluate_retrieval_metrics(model, device, index, group_families, k=10)

        eval_time = time.time() - start_time
        metrics = {
            **retrieval_metrics,
            'n_queries': len(self.eval_queries),
            'n_groups': len(sample_groups),
            'eval_time_sec': eval_time,
        }

        logger.info(
            f"[Tier 1] complete in {eval_time:.1f}s: "
            f"R@1={metrics['recall@1']:.3f}, R@5={metrics['recall@5']:.3f}, "
            f"R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}"
        )
        return metrics

    def evaluate_tier2(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        start_time = time.time()
        logger.info(f"[Tier 2] starting sampled evaluation ({self.n_samples_tier2} groups)...")
        if self._tier2_sample_cache is None:
            self._tier2_sample_cache = self.db.sample_groups(
                n=self.n_samples_tier2, seed=self.seed, exclude_curated=True
            )
        sample_groups = self._tier2_sample_cache
        group_embeddings, group_families = self._embed_groups(model, device, sample_groups)
        index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(group_embeddings)
        index.add(group_embeddings)
        recall_1, recall_5, recall_10 = [], [], []
        family_purity_1, family_purity_5, family_purity_10 = [], [], []
        for query_data in self.eval_queries:
            query_expr = query_data['expr']
            query_family = query_data.get('family', 'unknown')
            try:
                query_emb = self._embed_single_expr(model, device, query_expr)
                faiss.normalize_L2(query_emb)
                scores, indices = index.search(query_emb, k=10)
                retrieved_families = [group_families[idx] for idx in indices[0] if idx != -1]
                recall_1.append(1.0 if retrieved_families[:1].count(query_family) > 0 else 0.0)
                recall_5.append(1.0 if retrieved_families[:5].count(query_family) > 0 else 0.0)
                recall_10.append(1.0 if retrieved_families[:10].count(query_family) > 0 else 0.0)
                # family purity: fraction of top-k with same family as query
                family_purity_1.append(retrieved_families[:1].count(query_family) / 1)
                family_purity_5.append(retrieved_families[:5].count(query_family) / 5)
                family_purity_10.append(retrieved_families[:10].count(query_family) / 10)

            except Exception as e:
                logger.warning(f"failed to evaluate query '{query_expr}': {e}")
                continue

        intra_sim, inter_sim = self._compute_family_separation(group_embeddings, group_families)
        eval_time = time.time() - start_time
        metrics = {
            'recall@1': np.mean(recall_1),
            'recall@5': np.mean(recall_5),
            'recall@10': np.mean(recall_10),
            'family_purity@1': np.mean(family_purity_1),
            'family_purity@5': np.mean(family_purity_5),
            'family_purity@10': np.mean(family_purity_10),
            'intra_family_sim': intra_sim,
            'inter_family_sim': inter_sim,
            'separation_margin': intra_sim - inter_sim,
            'n_queries': len(self.eval_queries),
            'n_groups': len(sample_groups),
            'eval_time_sec': eval_time,
        }
        logger.info(
            f"[Tier 2] complete in {eval_time:.1f}s: "
            f"R@5={metrics['recall@5']:.3f}, purity@5={metrics['family_purity@5']:.3f}, "
            f"separation={metrics['separation_margin']:.3f}"
        )
        return metrics

    def evaluate_tier3(self, model: nn.Module, device: torch.device, iteration: int) -> Dict[str, any]:
        start_time = time.time()
        logger.info("[Tier 3] starting comprehensive evaluation (full database)...")
        if self._all_groups_cache is None:
            self._all_groups_cache = []
            offset = 0
            while True:
                batch = self.db.get_all_groups(limit=5000, offset=offset, exclude_curated=True)
                if not batch:
                    break
                self._all_groups_cache.extend(batch)
                offset += 5000
            logger.info(f"cached {len(self._all_groups_cache)} groups from database")
        all_groups = self._all_groups_cache
        group_embeddings, group_families = self._embed_groups(model, device, all_groups)
        index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(group_embeddings)
        index.add(group_embeddings)
        retrieval_metrics = self._evaluate_retrieval_metrics(model, device, index, group_families, k=10)
        analogy_metrics = self._evaluate_analogies(model, device, index, all_groups, k=10)
        intra_sim, inter_sim = self._compute_family_separation(group_embeddings, group_families)
        search_examples_text = self._generate_search_examples(model, device, index, all_groups, group_families)
        viz_path = None
        if self.checkpoint_dir:
            viz_path = self._generate_tsne_visualization(group_embeddings, group_families, iteration)

        eval_time = time.time() - start_time
        metrics = {
            **retrieval_metrics,
            **analogy_metrics,
            'intra_family_sim': intra_sim,
            'inter_family_sim': inter_sim,
            'separation_margin': intra_sim - inter_sim,
            'n_queries': len(self.eval_queries),
            'n_groups': len(all_groups),
            'eval_time_sec': eval_time,
        }
        log_msg = (
            f"[Tier 3] complete in {eval_time:.1f}s: "
            f"R@1={metrics['recall@1']:.3f}, R@5={metrics['recall@5']:.3f}, "
            f"R@10={metrics['recall@10']:.3f}, MRR={metrics['mrr']:.3f}, "
            f"separation={metrics['separation_margin']:.3f}"
        )
        if 'analogy_accuracy@5' in metrics:
            log_msg += f", analogy@5={metrics['analogy_accuracy@5']:.3f}"
        logger.info(log_msg)

        return {
            'metrics': metrics,
            'search_examples': search_examples_text,
            'visualization_path': str(viz_path) if viz_path else None,
        }

    def _evaluate_retrieval_metrics(self, model: nn.Module, device: torch.device, index: faiss.Index,
                                    group_families: List[str], k: int = 10) -> Dict[str, float]:
        recall_1, recall_5, recall_10 = [], [], []
        mrr_scores, map_scores = [], []
        for query_data in self.eval_queries:
            query_expr = query_data['expr']
            query_family = query_data.get('family', 'unknown')
            try:
                query_emb = self._embed_single_expr(model, device, query_expr)
                faiss.normalize_L2(query_emb)
                scores, indices = index.search(query_emb, k=k)
                retrieved_families = [group_families[idx] for idx in indices[0] if idx != -1]
                recall_1.append(1.0 if retrieved_families[:1].count(query_family) > 0 else 0.0)
                recall_5.append(1.0 if retrieved_families[:5].count(query_family) > 0 else 0.0)
                recall_10.append(1.0 if retrieved_families[:10].count(query_family) > 0 else 0.0)
                try:
                    first_correct_rank = retrieved_families.index(query_family) + 1
                    mrr_scores.append(1.0 / first_correct_rank)
                except ValueError:
                    mrr_scores.append(0.0)

                precisions = []
                for k_iter in range(1, len(retrieved_families) + 1):
                    if retrieved_families[k_iter-1] == query_family:
                        precision_at_k = sum(1 for f in retrieved_families[:k_iter] if f == query_family) / k_iter
                        precisions.append(precision_at_k)
                map_scores.append(np.mean(precisions) if precisions else 0.0)
            except Exception as e:
                logger.warning(f"failed to evaluate query '{query_expr}': {e}")
                continue
        return {
            'recall@1': float(np.mean(recall_1)),
            'recall@5': float(np.mean(recall_5)),
            'recall@10': float(np.mean(recall_10)),
            'mrr': float(np.mean(mrr_scores)),
            'map': float(np.mean(map_scores)),
        }

    def _evaluate_analogies(self, model: nn.Module, device: torch.device, index: faiss.Index,
                            groups: List[IntegrandGroup], k: int = 10) -> Dict[str, float]:
        if not self.analogy_queries:
            return {}
        accuracy_1, accuracy_5, accuracy_10 = [], [], []
        canonical_exprs = [g.integrand_canonical for g in groups]
        for analogy in self.analogy_queries:
            A, B, C, D_true = analogy['A'], analogy['B'], analogy['C'], analogy['D']
            try:
                emb_A = self._embed_single_expr(model, device, A)
                emb_B = self._embed_single_expr(model, device, B)
                emb_C = self._embed_single_expr(model, device, C)
                target_emb = emb_C + (emb_B - emb_A)
                faiss.normalize_L2(target_emb)
                scores, indices = index.search(target_emb, k=k)
                retrieved_exprs = [canonical_exprs[idx] for idx in indices[0] if idx != -1]
                D_true_sympified = str(sp.sympify(D_true))
                retrieved_sympified = [str(sp.sympify(expr)) for expr in retrieved_exprs]
                accuracy_1.append(1.0 if D_true_sympified in retrieved_sympified[:1] else 0.0)
                accuracy_5.append(1.0 if D_true_sympified in retrieved_sympified[:5] else 0.0)
                accuracy_10.append(1.0 if D_true_sympified in retrieved_sympified[:10] else 0.0)
            except Exception as e:
                logger.warning(f"failed to evaluate analogy {A}:{B}::{C}:{D_true}: {e}")
                continue
        if not accuracy_1:
            return {}
        return {
            'analogy_accuracy@1': float(np.mean(accuracy_1)),
            'analogy_accuracy@5': float(np.mean(accuracy_5)),
            'analogy_accuracy@10': float(np.mean(accuracy_10)),
        }

    def _embed_groups(self, model: nn.Module, device: torch.device, groups: List[IntegrandGroup]) -> Tuple[np.ndarray, List[str]]:
        model.eval()
        canonical_integrands = [g.integrand_canonical for g in groups]
        families = [g.integrand_family or 'unknown' for g in groups]
        # convert to prefix notation
        prefix_exprs = []
        for expr_str in canonical_integrands:
            try:
                expr = sp.sympify(expr_str)
                prefix = sympy_to_prefix(expr)
                prefix_exprs.append(prefix)
            except Exception as e:
                logger.warning(f"failed to convert '{expr_str}' to prefix: {e}")
                prefix_exprs.append("x")  # fallback
        # tokenize (use model's max_seq_len)
        max_len = getattr(model, 'max_seq_len', 128)
        token_ids_list = []
        for prefix in prefix_exprs:
            try:
                token_ids = self.tokenizer.encode(prefix, max_len=max_len)
                token_ids_list.append(token_ids)
            except Exception as e:
                logger.warning(f"tokenization failed for '{prefix}': {e}")
                token_ids_list.append(self.tokenizer.encode("x", max_len=max_len))
        # embed in batches
        batch_size = 256
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(token_ids_list), batch_size):
                batch_tokens = token_ids_list[i:i + batch_size]
                token_tensor = torch.tensor(batch_tokens, dtype=torch.long).to(device)
                mask = (token_tensor == self.tokenizer.PAD_ID)
                hidden = model(token_tensor, mask=mask)
                emb = model.mean_pool(hidden, mask=mask)
                all_embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return embeddings, families

    def _embed_single_expr(self, model: nn.Module, device: torch.device, expr_str: str) -> np.ndarray:
        model.eval()
        expr = sp.sympify(expr_str)
        prefix = sympy_to_prefix(expr)
        max_len = getattr(model, 'max_seq_len', 128)
        token_ids = self.tokenizer.encode(prefix, max_len=max_len)
        token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        mask = (token_tensor == self.tokenizer.PAD_ID)
        with torch.no_grad():
            hidden = model(token_tensor, mask=mask)
            emb = model.mean_pool(hidden, mask=mask)
        return emb.cpu().numpy().astype(np.float32)

    def _compute_family_separation(self, embeddings: np.ndarray, families: List[str]) -> Tuple[float, float]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        sim_matrix = embeddings_norm @ embeddings_norm.T
        intra_sims, inter_sims = [], []
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                if families[i] == families[j]:
                    intra_sims.append(sim_matrix[i, j])
                else:
                    inter_sims.append(sim_matrix[i, j])
        intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0
        inter_sim = float(np.mean(inter_sims)) if inter_sims else 0.0
        return intra_sim, inter_sim

    def _generate_search_examples(self, model: nn.Module, device: torch.device, index: faiss.Index,
                                  groups: List[IntegrandGroup], families: List[str]) -> str:
        test_queries = ["sin(x)", "cos(x)", "exp(x)", "log(x)", "exp(log(x))", "x*sin(x)", "exp(-x^2)"]
        lines = ["\n" + "="*80]
        lines.append("SEARCH EXAMPLES (Top-10 Most Similar Integrands)")
        lines.append("="*80)
        for query_expr in test_queries:
            try:
                query_emb = self._embed_single_expr(model, device, query_expr)
                faiss.normalize_L2(query_emb)
                scores, indices = index.search(query_emb, k=10)
                lines.append(f"\nQuery: {query_expr}")
                lines.append("-" * 80)
                for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                    if idx == -1:
                        break
                    group = groups[idx]
                    family = families[idx]
                    lines.append(
                        f"{rank:2d}. [{score:.4f}] {group.integrand_canonical:40s} "
                        f"(family: {family}, instances: {len(group.definite_instances) + len(group.indefinite_instances)})"
                    )
            except Exception as e:
                lines.append(f"\nQuery: {query_expr} - FAILED: {e}")
        lines.append("="*80 + "\n")
        return "\n".join(lines)

    def _generate_tsne_visualization(self, embeddings: np.ndarray, families: List[str], iteration: int) -> Optional[Path]:
        try:
            n_samples = min(self.n_viz_samples, len(embeddings))
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sampled_embeddings = embeddings[indices]
            sampled_families = [families[i] for i in indices]
            output_path = self.checkpoint_dir / f'tsne_iter{iteration:06d}.png'
            plot_embedding_space(
                embeddings=sampled_embeddings,
                labels=sampled_families,
                output_path=output_path,
                method='tsne',
                perplexity=min(30, n_samples // 3),
                n_components=2
            )
            logger.info(f"saved t-SNE visualization: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"failed to generate t-SNE visualization: {e}")
            return None
