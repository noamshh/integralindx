"""
Test trained model on analogy queries.

This script uses cached embeddings from data/embeddings/ if available (much faster).
Use --no-cache to force rebuilding embeddings from scratch.

Usage:
    # basic usage (uses cached embeddings if available)
    python scripts/test_analogies.py --checkpoint data/models/checkpoints/ii-cl-19m_v1.0/best_model.pt

    # verbose output (shows all predictions)
    python scripts/test_analogies.py --checkpoint data/models/checkpoints/ii-cl-19m_v1.0/best_model.pt --verbose

    # force rebuild embeddings (ignore cache)
    python scripts/test_analogies.py --checkpoint data/models/checkpoints/ii-cl-19m_v1.0/best_model.pt --no-cache

    # test baseline model
    python scripts/test_analogies.py --checkpoint data/models/checkpoints/ii-cl-400k_v1.0/best_model.pt --model ii-cl-400k

    # save detailed results to JSON
    python scripts/test_analogies.py --checkpoint data/models/checkpoints/ii-cl-19m_v1.0/best_model.pt --output results.json
"""
import argparse
import json
import logging
from pathlib import Path
import torch
import numpy as np
import faiss
import sympy as sp
from omegaconf import OmegaConf

from src.models.egen.contrastive_model import Encoder
from src.models.egen.tokenizer import Tokenizer
from src.database.integral_db import IntegralDatabase
from src.utils.prefix_notation import sympy_to_prefix
from src.utils.paths import get_paths
from src.search.embedding_cache import cache_exists, load_embedding_cache

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, model_name: str, device: torch.device):
    """load trained model from checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"loaded checkpoint from {checkpoint_path}")

    paths = get_paths()
    model_config_path = paths['project_root'] / 'config' / 'model' / f'{model_name}.yaml'
    if not model_config_path.exists():
        raise FileNotFoundError(f"model config not found: {model_config_path}")

    model_cfg = OmegaConf.load(model_config_path)
    encoder_cfg = model_cfg.encoder

    tokenizer = Tokenizer()
    vocab_size = len(tokenizer)

    model = Encoder(
        vocab_size=vocab_size,
        dim=encoder_cfg.dim,
        num_layers=encoder_cfg.num_layers,
        num_heads=encoder_cfg.num_heads,
        feedforward_dim=encoder_cfg.feedforward_dim,
        max_seq_len=encoder_cfg.max_seq_len,
        dropout=encoder_cfg.dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"loaded model: {model_cfg.name} ({encoder_cfg.dim}D, {encoder_cfg.num_layers}L)")

    return model, tokenizer, model_cfg


def embed_expr(model: torch.nn.Module, tokenizer: Tokenizer, expr_str: str, device: torch.device) -> np.ndarray:
    """embed a single expression"""
    expr = sp.sympify(expr_str)
    prefix = sympy_to_prefix(expr)
    max_len = model.max_seq_len
    token_ids = tokenizer.encode(prefix, max_len=max_len)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = (token_tensor == tokenizer.PAD_ID)

    with torch.no_grad():
        hidden = model(token_tensor, mask=mask)
        emb = model.mean_pool(hidden, mask=mask)

    return emb.cpu().numpy().astype(np.float32)


def load_or_build_index(model: torch.nn.Module, tokenizer: Tokenizer, db: IntegralDatabase,
                         device: torch.device, embedder_name: str, use_cache: bool = True):
    """load cached embeddings or build from scratch

    Args:
        model: trained encoder model
        tokenizer: tokenizer
        db: integral database
        device: torch device
        embedder_name: name of embedder (e.g., 'ii-cl-19m_v1.0')
        use_cache: if True, try to load from cache first

    Returns:
        tuple of (index, canonical_exprs, all_groups)
    """
    # try to load from cache
    if use_cache and cache_exists(embedder_name):
        logger.info(f"found cached embeddings for '{embedder_name}'")
        logger.info("loading from cache (this is much faster than rebuilding)...")
        try:
            search_engine = load_embedding_cache(embedder_name, embedder=None)
            canonical_exprs = [g.integrand_canonical for g in search_engine.groups]
            logger.info(f"loaded {len(search_engine.groups)} groups from cache")
            return search_engine.index, canonical_exprs, search_engine.groups
        except Exception as e:
            logger.warning(f"failed to load cache: {e}")
            logger.warning("falling back to building index from scratch...")

    # build from scratch
    logger.info("building FAISS index from scratch (no cache found)...")
    logger.info("loading integrand groups from database...")
    all_groups = []
    offset = 0
    while True:
        batch = db.get_all_groups(limit=5000, offset=offset, exclude_curated=True)
        if not batch:
            break
        all_groups.extend(batch)
        offset += 5000

    logger.info(f"loaded {len(all_groups)} groups, building embeddings...")

    # embed all canonical integrands
    canonical_exprs = [g.integrand_canonical for g in all_groups]
    embeddings_list = []

    for i, expr_str in enumerate(canonical_exprs):
        if i % 1000 == 0:
            logger.info(f"embedded {i}/{len(canonical_exprs)} groups...")
        try:
            expr = sp.sympify(expr_str)
            prefix = sympy_to_prefix(expr)
            max_len = model.max_seq_len
            token_ids = tokenizer.encode(prefix, max_len=max_len)
            token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
            mask = (token_tensor == tokenizer.PAD_ID)

            with torch.no_grad():
                hidden = model(token_tensor, mask=mask)
                emb = model.mean_pool(hidden, mask=mask)

            embeddings_list.append(emb.cpu().numpy())
        except Exception as e:
            logger.warning(f"failed to embed '{expr_str}': {e}")
            # fallback to x
            emb = embed_expr(model, tokenizer, "x", device)
            embeddings_list.append(emb)

    embeddings = np.vstack(embeddings_list).astype(np.float32)

    # build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    logger.info(f"built FAISS index with {len(all_groups)} groups")
    return index, canonical_exprs, all_groups


def evaluate_analogies(model: torch.nn.Module, tokenizer: Tokenizer, analogy_queries: list,
                       index: faiss.Index, canonical_exprs: list, device: torch.device,
                       verbose: bool = False, k: int = 10):
    """evaluate analogy queries using vector arithmetic: D = C + (B - A)"""
    logger.info(f"evaluating {len(analogy_queries)} analogy queries...")

    results = []
    accuracy_1, accuracy_5, accuracy_10 = [], [], []

    for i, analogy in enumerate(analogy_queries):
        A, B, C, D_true = analogy['A'], analogy['B'], analogy['C'], analogy['D']
        category = analogy.get('category', 'unknown')
        source = analogy.get('source', 'unknown')

        try:
            # embed A, B, C
            emb_A = embed_expr(model, tokenizer, A, device)
            emb_B = embed_expr(model, tokenizer, B, device)
            emb_C = embed_expr(model, tokenizer, C, device)

            # compute target: D = C + (B - A)
            target_emb = emb_C + (emb_B - emb_A)
            faiss.normalize_L2(target_emb)

            # search for nearest neighbors
            scores, indices = index.search(target_emb, k=k)
            retrieved_exprs = [canonical_exprs[idx] for idx in indices[0] if idx != -1]

            # sympify for comparison
            D_true_sympified = str(sp.sympify(D_true))
            retrieved_sympified = [str(sp.sympify(expr)) for expr in retrieved_exprs]

            # check accuracy
            hit_at_1 = D_true_sympified in retrieved_sympified[:1]
            hit_at_5 = D_true_sympified in retrieved_sympified[:5]
            hit_at_10 = D_true_sympified in retrieved_sympified[:10]

            accuracy_1.append(1.0 if hit_at_1 else 0.0)
            accuracy_5.append(1.0 if hit_at_5 else 0.0)
            accuracy_10.append(1.0 if hit_at_10 else 0.0)

            result = {
                'id': i,
                'A': A,
                'B': B,
                'C': C,
                'D_true': D_true,
                'category': category,
                'source': source,
                'hit@1': hit_at_1,
                'hit@5': hit_at_5,
                'hit@10': hit_at_10,
                'top10_retrieved': retrieved_exprs[:10],
                'top10_scores': scores[0][:10].tolist(),
            }
            results.append(result)

            if verbose:
                status = "✓" if hit_at_5 else "✗"
                print(f"\n{status} [{i+1}/{len(analogy_queries)}] {A} : {B} :: {C} : ?")
                print(f"   Category: {category}")
                print(f"   Expected: {D_true}")
                print(f"   Predicted (top-10):")
                for rank, (expr, score) in enumerate(zip(retrieved_exprs[:10], scores[0][:10]), 1):
                    match_marker = "✓" if str(sp.sympify(expr)) == D_true_sympified else " "
                    print(f"      {rank:2d}. [{score:.4f}] {match_marker} {expr}")
                if not hit_at_10:
                    print(f"   ⚠ Expected answer not in top-10!")

        except Exception as e:
            logger.warning(f"failed to evaluate analogy {A}:{B}::{C}:{D_true}: {e}")
            results.append({
                'id': i,
                'A': A,
                'B': B,
                'C': C,
                'D_true': D_true,
                'category': category,
                'source': source,
                'error': str(e),
            })

    # aggregate metrics
    overall_metrics = {
        'accuracy@1': float(np.mean(accuracy_1)) if accuracy_1 else 0.0,
        'accuracy@5': float(np.mean(accuracy_5)) if accuracy_5 else 0.0,
        'accuracy@10': float(np.mean(accuracy_10)) if accuracy_10 else 0.0,
        'n_queries': len(analogy_queries),
        'n_succeeded': len(accuracy_1),
    }

    # per-category metrics
    category_metrics = {}
    for result in results:
        if 'error' in result:
            continue
        cat = result['category']
        if cat not in category_metrics:
            category_metrics[cat] = {'accuracy@1': [], 'accuracy@5': [], 'accuracy@10': []}
        category_metrics[cat]['accuracy@1'].append(result['hit@1'])
        category_metrics[cat]['accuracy@5'].append(result['hit@5'])
        category_metrics[cat]['accuracy@10'].append(result['hit@10'])

    for cat, metrics in category_metrics.items():
        category_metrics[cat] = {
            'accuracy@1': float(np.mean(metrics['accuracy@1'])),
            'accuracy@5': float(np.mean(metrics['accuracy@5'])),
            'accuracy@10': float(np.mean(metrics['accuracy@10'])),
            'n_queries': len(metrics['accuracy@1']),
        }

    return overall_metrics, category_metrics, results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"using device: {device}")

    # load model
    checkpoint_path = Path(args.checkpoint)
    model, tokenizer, model_cfg = load_model(checkpoint_path, args.model, device)

    # determine embedder name from model config
    # format: model_name_version (e.g., "ii-cl-19m_v1.0")
    embedder_name = f"{model_cfg.name}_{model_cfg.version}"
    logger.info(f"embedder name: {embedder_name}")

    # load analogy queries
    paths = get_paths()
    analogy_path = Path(args.analogy_queries)
    if not analogy_path.exists():
        raise FileNotFoundError(f"analogy queries not found: {analogy_path}")

    with open(analogy_path) as f:
        data = json.load(f)
        analogy_queries = data['analogies']
    logger.info(f"loaded {len(analogy_queries)} analogy queries")

    # load or build database index
    db = IntegralDatabase(db_path=Path(args.db_path), fallback_to_jsonl=False)
    use_cache = not args.no_cache
    index, canonical_exprs, all_groups = load_or_build_index(
        model, tokenizer, db, device, embedder_name, use_cache=use_cache
    )

    # evaluate analogies
    overall_metrics, category_metrics, results = evaluate_analogies(
        model, tokenizer, analogy_queries, index, canonical_exprs, device,
        verbose=args.verbose, k=10
    )

    # print results
    logger.info("=" * 80)
    logger.info("ANALOGY EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Overall Accuracy:")
    logger.info(f"  Accuracy@1:  {overall_metrics['accuracy@1']:.3f} ({int(overall_metrics['accuracy@1'] * overall_metrics['n_succeeded'])}/{overall_metrics['n_succeeded']})")
    logger.info(f"  Accuracy@5:  {overall_metrics['accuracy@5']:.3f} ({int(overall_metrics['accuracy@5'] * overall_metrics['n_succeeded'])}/{overall_metrics['n_succeeded']})")
    logger.info(f"  Accuracy@10: {overall_metrics['accuracy@10']:.3f} ({int(overall_metrics['accuracy@10'] * overall_metrics['n_succeeded'])}/{overall_metrics['n_succeeded']})")
    logger.info("")
    logger.info("Per-Category Accuracy:")
    for cat, metrics in sorted(category_metrics.items()):
        logger.info(f"  {cat:30s}: @1={metrics['accuracy@1']:.3f}, @5={metrics['accuracy@5']:.3f}, @10={metrics['accuracy@10']:.3f} (n={metrics['n_queries']})")
    logger.info("=" * 80)

    # show example predictions (if not already shown in verbose mode)
    if not args.verbose:
        logger.info("")
        logger.info("EXAMPLE PREDICTIONS (first 5 queries):")
        logger.info("=" * 80)
        for result in results[:5]:
            if 'error' in result:
                continue
            status = "✓" if result['hit@5'] else "✗"
            print(f"\n{status} {result['A']} : {result['B']} :: {result['C']} : ?")
            print(f"   Expected: {result['D_true']}")
            print(f"   Predicted: {result['top10_retrieved'][0]} (score: {result['top10_scores'][0]:.4f})")
            if result['hit@1']:
                print(f"   ✓ Correct at rank 1!")
            elif result['hit@5']:
                # find rank
                D_true_sympified = str(sp.sympify(result['D_true']))
                for rank, expr in enumerate(result['top10_retrieved'], 1):
                    if str(sp.sympify(expr)) == D_true_sympified:
                        print(f"   ✓ Correct at rank {rank}")
                        break
            else:
                print(f"   ✗ Expected answer not in top-10")
        logger.info("=" * 80)
        logger.info("(Use --verbose to see all predictions)")
        logger.info("=" * 80)

    # save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            'checkpoint': str(checkpoint_path),
            'model': args.model,
            'overall_metrics': overall_metrics,
            'category_metrics': category_metrics,
            'results': results,
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"saved detailed results to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test trained model on analogy queries')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--model', type=str, default='ii-cl-19m', help='model config name (default: ii-cl-19m)')
    parser.add_argument('--training', type=str, default='gpu', help='training config name (default: gpu)')
    parser.add_argument('--analogy-queries', type=str, default='data/eval/analogy_queries.json', help='path to analogy queries JSON')
    parser.add_argument('--db-path', type=str, default='data/integral.db', help='path to SQLite database')
    parser.add_argument('--output', type=str, default=None, help='save detailed results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='print detailed results for each query')
    parser.add_argument('--no-cache', action='store_true', help='force rebuild embeddings instead of using cache')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main(args)
