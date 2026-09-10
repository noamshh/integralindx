"""
evaluate vector math analogies on embedding models using pre-built caches

evaluates A:B::C:D analogies where target_embedding = C + (B - A)

usage:
    # default: evaluate all embedders
    python -m scripts.evaluate_analogy

    # custom output directory
    python -m scripts.evaluate_analogy --output-dir evaluation_results/analogy
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import faiss
import sympy as sp

from src.database.integral_db import IntegralDatabase
from src.search.similarity_engine import IntegrandGroupSearch
from src.search import embedding_cache
from src.models.egen.contrastive_embedder import CLEmbedder
from src.utils.paths import get_paths
from src.utils.prefix_notation import sympy_to_prefix

logger = logging.getLogger(__name__)


def filter_valid_analogies(analogies: List[Dict], canonical_exprs: List[str]) -> Tuple[List[Dict], Dict]:
    """filter analogies to only include those where D exists in the dataset

    Args:
        analogies: list of analogy dicts with A, B, C, D keys
        canonical_exprs: list of canonical expressions in the dataset

    Returns:
        (filtered_analogies, filter_stats)
    """
    # sympify all canonical expressions for comparison
    canonical_sympified = set()
    for expr_str in canonical_exprs:
        try:
            canonical_sympified.add(str(sp.sympify(expr_str)))
        except Exception:
            pass

    valid_analogies = []
    for analogy in analogies:
        try:
            D_sympified = str(sp.sympify(analogy['D']))
            if D_sympified in canonical_sympified:
                valid_analogies.append(analogy)
        except Exception:
            pass

    filter_stats = {
        'total_analogies': len(analogies),
        'valid_analogies': len(valid_analogies),
        'filtered_out': len(analogies) - len(valid_analogies),
        'filter_rate': (len(analogies) - len(valid_analogies)) / len(analogies) if analogies else 0
    }

    logger.info(
        f"filtered analogies: {filter_stats['valid_analogies']}/{filter_stats['total_analogies']} valid "
        f"({filter_stats['filtered_out']} filtered out, {filter_stats['filter_rate']:.1%} filter rate)"
    )

    return valid_analogies, filter_stats


def embed_expression(embedder: Any, expr_str: str) -> np.ndarray:
    """embed a single expression using the embedder

    Args:
        embedder: embedder instance (CLEmbedder or BaselineEmbedder)
        expr_str: expression string

    Returns:
        embedding vector (1D numpy array)
    """
    try:
        # convert to sympy
        expr = sp.sympify(expr_str)

        # convert to prefix notation
        prefix = sympy_to_prefix(expr)

        # embed using embedder
        embedding = embedder.embed_single(prefix)

        return embedding.flatten()
    except Exception as e:
        logger.warning(f"failed to embed '{expr_str}': {e}")
        # return zero vector as fallback
        dim = getattr(embedder, 'embedding_dim', 384)
        return np.zeros(dim, dtype=np.float32)


def evaluate_analogies(embedder_name: str, embedder: Any, search_engine: IntegrandGroupSearch,
                       analogies: List[Dict], groups: List[Any], k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
    """evaluate analogies on a single embedder

    Args:
        embedder_name: name of embedder for logging
        embedder: embedder instance
        search_engine: pre-built search engine with FAISS index
        analogies: list of analogy dicts
        groups: list of IntegrandGroup objects from database
        k_values: k values for accuracy@k

    Returns:
        dict with metrics and detailed results
    """
    logger.info(f"evaluating {len(analogies)} analogies on {embedder_name}...")

    start_time = time.time()

    # get canonical expressions from groups (passed in, not from search_engine which is empty)
    canonical_exprs = [g.integrand_canonical for g in groups]

    # sympify all canonical expressions for comparison
    canonical_sympified = []
    for expr_str in canonical_exprs:
        try:
            canonical_sympified.append(str(sp.sympify(expr_str)))
        except Exception:
            canonical_sympified.append(expr_str)

    # evaluate each analogy
    results = []
    accuracy = {k: [] for k in k_values}
    category_results = {}

    for analogy in analogies:
        A, B, C, D_true = analogy['A'], analogy['B'], analogy['C'], analogy['D']
        category = analogy.get('category', 'unknown')

        try:
            # compute target embedding: C + (B - A)
            emb_A = embed_expression(embedder, A)
            emb_B = embed_expression(embedder, B)
            emb_C = embed_expression(embedder, C)
            target_emb = emb_C + (emb_B - emb_A)

            # normalize for cosine similarity
            target_emb = target_emb.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(target_emb)

            # search FAISS index
            scores, indices = search_engine.index.search(target_emb, k=max(k_values))

            # get retrieved expressions
            retrieved_exprs = []
            for idx in indices[0]:
                if idx != -1 and idx < len(canonical_sympified):
                    retrieved_exprs.append(canonical_sympified[idx])

            # check if D_true is in top-k
            D_true_sympified = str(sp.sympify(D_true))

            for k in k_values:
                is_correct = D_true_sympified in retrieved_exprs[:k]
                accuracy[k].append(1.0 if is_correct else 0.0)

            # record per-category results
            if category not in category_results:
                category_results[category] = {k: [] for k in k_values}
            for k in k_values:
                is_correct = D_true_sympified in retrieved_exprs[:k]
                category_results[category][k].append(1.0 if is_correct else 0.0)

            # store detailed result
            results.append({
                'A': A,
                'B': B,
                'C': C,
                'D_true': D_true,
                'D_predicted': canonical_exprs[indices[0][0]] if indices[0][0] != -1 else 'N/A',
                'rank_of_D': retrieved_exprs.index(D_true_sympified) + 1 if D_true_sympified in retrieved_exprs else -1,
                'category': category,
                'success': D_true_sympified in retrieved_exprs[:10]
            })

        except Exception as e:
            logger.warning(f"failed to evaluate analogy {A}:{B}::{C}:{D_true}: {e}")
            # record failure
            for k in k_values:
                accuracy[k].append(0.0)
            if category not in category_results:
                category_results[category] = {k: [] for k in k_values}
            for k in k_values:
                category_results[category][k].append(0.0)

    eval_time = time.time() - start_time

    # compute overall metrics
    metrics = {
        f'accuracy@{k}': float(np.mean(accuracy[k])) if accuracy[k] else 0.0
        for k in k_values
    }
    metrics['eval_time_sec'] = eval_time
    metrics['n_analogies'] = len(analogies)

    # compute per-category metrics
    per_category_metrics = {}
    for category, cat_results in category_results.items():
        per_category_metrics[category] = {
            f'accuracy@{k}': float(np.mean(cat_results[k])) if cat_results[k] else 0.0
            for k in k_values
        }
        per_category_metrics[category]['n_analogies'] = len(cat_results[k_values[0]])

    logger.info(
        f"{embedder_name} results: "
        f"acc@1={metrics['accuracy@1']:.3f}, acc@5={metrics['accuracy@5']:.3f}, acc@10={metrics['accuracy@10']:.3f}"
    )

    return {
        'metrics': metrics,
        'per_category_metrics': per_category_metrics,
        'detailed_results': results
    }


def generate_analogy_examples(embedder_name: str, results: List[Dict], n_examples: int = 10) -> str:
    """generate human-readable examples of analogy results

    Args:
        embedder_name: name of embedder
        results: list of detailed analogy results
        n_examples: number of examples to show

    Returns:
        formatted text
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"ANALOGY EVALUATION EXAMPLES ({embedder_name})")
    lines.append("=" * 80)

    # show mix of successes and failures
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]

    n_success = min(n_examples // 2, len(successes))
    n_failure = min(n_examples - n_success, len(failures))

    # select diverse examples
    if successes:
        success_indices = np.linspace(0, len(successes) - 1, n_success, dtype=int)
        selected_successes = [successes[i] for i in success_indices]
    else:
        selected_successes = []

    if failures:
        failure_indices = np.linspace(0, len(failures) - 1, n_failure, dtype=int)
        selected_failures = [failures[i] for i in failure_indices]
    else:
        selected_failures = []

    # show successes
    if selected_successes:
        lines.append("\nSUCCESSFUL ANALOGIES:")
        lines.append("-" * 80)
        for r in selected_successes:
            status = "✓" if r['rank_of_D'] != -1 else "✗"
            rank_str = f"rank {r['rank_of_D']}" if r['rank_of_D'] != -1 else "not found"
            lines.append(f"{status} {r['A']} : {r['B']} :: {r['C']} : {r['D_true']}")
            lines.append(f"   Category: {r['category']}, {rank_str}")
            lines.append("")

    # show failures
    if selected_failures:
        lines.append("\nFAILED ANALOGIES:")
        lines.append("-" * 80)
        for r in selected_failures:
            lines.append(f"✗ {r['A']} : {r['B']} :: {r['C']} : {r['D_true']}")
            lines.append(f"   Category: {r['category']}")
            lines.append(f"   Predicted: {r['D_predicted']}")
            lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def main(args):
    paths = get_paths()

    # load analogies
    if args.analogy_file:
        analogies_path = Path(args.analogy_file)
    else:
        analogies_path = paths['data']['root'] / 'eval' / 'analogy_queries_focused.json'

    if not analogies_path.exists():
        raise FileNotFoundError(f"analogies file not found: {analogies_path}")

    with open(analogies_path) as f:
        data = json.load(f)
        all_analogies = data['analogies']

    logger.info(f"loaded {len(all_analogies)} analogies from {analogies_path}")

    # connect to database
    db_path = paths['data']['root'] / 'integral.db'
    db = IntegralDatabase(db_path=db_path, fallback_to_jsonl=False)
    logger.info(f"connected to database: {db_path}")

    # load embedders
    embedders_to_eval = {}

    # 1. E-Gen (ii-cl-19m_prod)
    logger.info("loading ii-cl-19m_prod embedding cache...")
    cache_info = embedding_cache.get_cache_info('ii-cl-19m_prod')
    checkpoint_path = Path(cache_info['checkpoint_path'])
    config_path = Path(cache_info['config_path']) if cache_info.get('config_path') else None
    logger.info(f"loading E-Gen model from: {checkpoint_path}")
    egen_embedder = CLEmbedder.load(checkpoint_path, config_path=config_path)
    egen_search_engine = IntegrandGroupSearch.load_cache('ii-cl-19m_prod', egen_embedder, db)
    embedders_to_eval['ii-cl-19m'] = (egen_embedder, egen_search_engine)

    # 2. TF-IDF
    try:
        logger.info("loading TF-IDF embedding cache...")
        tfidf_search_engine = IntegrandGroupSearch.load_cache('tfidf', embedder=None, database=db)
        embedders_to_eval['tf-idf'] = (tfidf_search_engine.embedder, tfidf_search_engine)
    except Exception as e:
        logger.warning(f"failed to load TF-IDF: {e}")

    # 3. Sentence-BERT
    try:
        logger.info("loading Sentence-BERT embedding cache...")
        sbert_search_engine = IntegrandGroupSearch.load_cache('sentence_bert', embedder=None, database=db)
        embedders_to_eval['sentence-bert'] = (sbert_search_engine.embedder, sbert_search_engine)
    except Exception as e:
        logger.warning(f"failed to load Sentence-BERT: {e}")

    # load groups from database (search_engine.groups is empty when loading from cache)
    logger.info("loading all groups from database for analogy filtering...")
    all_groups = db.get_all_groups(limit=20000, exclude_curated=True)
    canonical_exprs = [g.integrand_canonical for g in all_groups]
    logger.info(f"loaded {len(all_groups)} groups for filtering")
    valid_analogies, filter_stats = filter_valid_analogies(all_analogies, canonical_exprs)

    # evaluate each embedder
    all_results = {}
    for embedder_name, (embedder, search_engine) in embedders_to_eval.items():
        eval_results = evaluate_analogies(
            embedder_name=embedder_name,
            embedder=embedder,
            search_engine=search_engine,
            analogies=valid_analogies,
            groups=all_groups
        )
        all_results[embedder_name] = eval_results

    # output results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = paths['data']['root'] / 'evaluation_results' / 'ii-cl-19m_prod'

    output_dir.mkdir(parents=True, exist_ok=True)

    # save metrics JSON
    output_data = {
        'analogies_file': str(analogies_path),
        'filter_stats': filter_stats,
        'embedders': {
            name: {
                'metrics': results['metrics'],
                'per_category_metrics': results['per_category_metrics']
            }
            for name, results in all_results.items()
        }
    }

    output_path = output_dir / 'analogy_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"saved analogy results to {output_path}")

    # generate and save qualitative examples for each embedder
    for embedder_name, results in all_results.items():
        examples_text = generate_analogy_examples(embedder_name, results['detailed_results'])
        examples_path = output_dir / f'analogy_examples_{embedder_name}.txt'
        with open(examples_path, 'w') as f:
            f.write(examples_text)
        logger.info(f"saved qualitative examples for {embedder_name} to {examples_path}")

    # print summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALOGY EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Filter stats: {filter_stats['valid_analogies']}/{filter_stats['total_analogies']} valid")

    for embedder_name, results in all_results.items():
        metrics = results['metrics']
        logger.info(f"\n{embedder_name}:")
        logger.info(f"  Accuracy@1:  {metrics['accuracy@1']:.4f}")
        logger.info(f"  Accuracy@5:  {metrics['accuracy@5']:.4f}")
        logger.info(f"  Accuracy@10: {metrics['accuracy@10']:.4f}")

    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate vector math analogies on embedding models')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='output directory (default: data/evaluation_results/ii-cl-19m_prod)')
    parser.add_argument('--analogy-file', type=str, default=None,
                        help='path to analogy queries JSON file (default: data/eval/analogy_queries_focused.json)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main(args)
