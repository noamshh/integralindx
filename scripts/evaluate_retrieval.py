"""
evaluate retrieval performance on database using pre-built caches

uses:
- ii-cl-19m_prod cache (E-Gen trained model)
- tfidf cache (baseline)

usage:
    # default: compare to baselines
    python -m scripts.evaluate_retrieval

    # skip baseline comparison
    python -m scripts.evaluate_retrieval --no-baselines

    # custom output directory
    python -m scripts.evaluate_retrieval --output-dir evaluation_results/retrieval
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.database.integral_db import IntegralDatabase
from src.models.egen.contrastive_embedder import CLEmbedder
from src.search import embedding_cache
from src.utils.paths import get_paths

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(queries: List[Dict], results: List[List[Dict[str, Any]]],
                                k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """compute standard retrieval metrics

    Args:
        queries: list of query dicts with 'expr' and 'family' keys
        results: list of retrieval results (each is list of IntegrandGroup)
        k_values: k values for recall@k

    Returns:
        dict of metric name -> value
    """
    metrics = {}

    # recall@k: fraction of queries where correct family appears in top-k
    for k in k_values:
        correct = 0
        for query, result_list in zip(queries, results):
            query_family = query.get('family', 'unknown')
            top_k_families = [g.get('integrand_family') or 'unknown' for g in result_list[:k]]
            if query_family in top_k_families:
                correct += 1
        metrics[f'recall@{k}'] = correct / len(queries)

    # MRR (mean reciprocal rank)
    mrr_scores = []
    for query, result_list in zip(queries, results):
        query_family = query.get('family', 'unknown')
        families = [g.get('integrand_family') or 'unknown' for g in result_list]
        try:
            rank = families.index(query_family) + 1
            mrr_scores.append(1.0 / rank)
        except ValueError:
            mrr_scores.append(0.0)
    metrics['mrr'] = float(np.mean(mrr_scores))

    # MAP (mean average precision)
    map_scores = []
    for query, result_list in zip(queries, results):
        query_family = query.get('family', 'unknown')
        families = [g.get('integrand_family') or 'unknown' for g in result_list]

        # compute average precision for this query
        num_relevant = 0
        sum_precisions = 0.0
        for k in range(1, len(families) + 1):
            if families[k-1] == query_family:
                num_relevant += 1
                precision_at_k = num_relevant / k
                sum_precisions += precision_at_k

        if num_relevant > 0:
            map_scores.append(sum_precisions / num_relevant)
        else:
            map_scores.append(0.0)

    metrics['map'] = float(np.mean(map_scores))

    # NDCG (normalized discounted cumulative gain) at k=10
    ndcg_scores = []
    for query, result_list in zip(queries, results):
        query_family = query.get('family', 'unknown')
        families = [g.get('integrand_family') or 'unknown' for g in result_list[:10]]

        # relevance: 1 if same family, 0 otherwise
        relevance = [1 if f == query_family else 0 for f in families]

        # DCG
        dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance))

        # ideal DCG (all relevant docs at top)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    metrics['ndcg@10'] = float(np.mean(ndcg_scores))

    return metrics


def compute_per_family_metrics(queries: List[Dict], results: List[List[Dict[str, Any]]]) -> Dict[str, Dict]:
    """compute metrics broken down by query family"""
    family_stats = {}

    for query, result_list in zip(queries, results):
        query_family = query.get('family', 'unknown')

        if query_family not in family_stats:
            family_stats[query_family] = {
                'count': 0,
                'recall@1': 0,
                'recall@5': 0,
                'recall@10': 0,
                'mrr_sum': 0.0,
            }

        stats = family_stats[query_family]
        stats['count'] += 1

        families = [g.get('integrand_family') or 'unknown' for g in result_list]

        # recall@k
        if query_family in families[:1]:
            stats['recall@1'] += 1
        if query_family in families[:5]:
            stats['recall@5'] += 1
        if query_family in families[:10]:
            stats['recall@10'] += 1

        # mrr
        try:
            rank = families.index(query_family) + 1
            stats['mrr_sum'] += 1.0 / rank
        except ValueError:
            pass

    # compute averages
    for family, stats in family_stats.items():
        count = stats['count']
        stats['recall@1'] = stats['recall@1'] / count
        stats['recall@5'] = stats['recall@5'] / count
        stats['recall@10'] = stats['recall@10'] / count
        stats['mrr'] = stats['mrr_sum'] / count

    return family_stats


def generate_qualitative_examples(queries: List[Dict], results: List[List[Dict[str, Any]]],
                                    n_examples: int = 5) -> str:
    """generate human-readable examples of retrieval results"""
    lines = []

    # select diverse examples
    selected_indices = np.linspace(0, len(queries) - 1, n_examples, dtype=int)

    for idx in selected_indices:
        query = queries[idx]
        result_list = results[idx][:10]  # top-10

        lines.append("\n" + "=" * 80)
        lines.append(f"Query: {query['expr']} (family: {query.get('family', 'unknown')})")
        lines.append("-" * 80)

        for rank, group in enumerate(result_list, 1):
            family = group.get('integrand_family') or 'unknown'
            n_instances = group.get('definite_count', 0) + group.get('indefinite_count', 0)
            match_indicator = "✓" if family == query.get('family', 'unknown') else " "
            lines.append(
                f"{rank:2d}. {match_indicator} [{family:20s}] {group.get('integrand_canonical', 'N/A'):40s} ({n_instances} instances)"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def evaluate_embedder(embedder_name: str, embedder: Any, db: IntegralDatabase,
                      queries: List[Dict], search_engine: Optional[Any] = None) -> Tuple[Dict[str, float], List[List[Dict[str, Any]]]]:
    """evaluate a single embedder on retrieval task

    Args:
        embedder_name: name of the embedder
        embedder: embedder instance (or None if using pre-built search_engine)
        db: database instance
        queries: list of query dicts
        search_engine: optional pre-built search engine (if provided, skips index building)
    """
    logger.info(f"evaluating {embedder_name}...")

    start_time = time.time()

    # build or use pre-built search engine
    if search_engine is None:
        from src.search.similarity_engine import IntegrandGroupSearch
        logger.info("loading all groups from database...")
        all_groups = db.get_all_groups(limit=20000, exclude_curated=True)
        logger.info(f"loaded {len(all_groups)} groups")

        logger.info("building FAISS index...")
        search_engine = IntegrandGroupSearch(embedding_dim=embedder.embedding_dim, database=db)
        search_engine.groups = all_groups
        search_engine.build_index(embedder, batch_size=100)
        logger.info(f"built index with {search_engine.index.ntotal} vectors")
    else:
        logger.info(f"using pre-built search engine with {search_engine.index.ntotal} vectors")

    # retrieve for each query
    all_results = []
    for query_data in queries:
        query_expr = query_data['expr']
        try:
            # search with k=20 to compute all metrics
            results = search_engine.search(query=query_expr, k=20)
            all_results.append(results)
        except Exception as e:
            logger.warning(f"failed to search for '{query_expr}': {e}")
            all_results.append([])  # empty results

    # compute metrics
    metrics = compute_retrieval_metrics(queries, all_results)
    per_family = compute_per_family_metrics(queries, all_results)

    eval_time = time.time() - start_time
    metrics['eval_time_sec'] = eval_time

    logger.info(
        f"{embedder_name} results: "
        f"R@1={metrics['recall@1']:.3f}, R@5={metrics['recall@5']:.3f}, R@10={metrics['recall@10']:.3f}, "
        f"MRR={metrics['mrr']:.3f}, MAP={metrics['map']:.3f}, NDCG@10={metrics['ndcg@10']:.3f}"
    )

    return metrics, all_results, per_family


def main(args):
    paths = get_paths()

    # load queries
    queries_path = paths['data']['root'] / 'eval' / 'topk_queries.json'
    if not queries_path.exists():
        raise FileNotFoundError(f"queries file not found: {queries_path}")

    with open(queries_path) as f:
        data = json.load(f)
        queries = data['queries']

    logger.info(f"loaded {len(queries)} evaluation queries from {queries_path}")

    # connect to database
    db_path = paths['data']['root'] / 'integral.db'
    db = IntegralDatabase(db_path=db_path, fallback_to_jsonl=False)
    logger.info(f"connected to database: {db_path}")

    # load E-Gen cache (ii-cl-19m_prod) - following webapp pattern
    logger.info("loading ii-cl-19m_prod embedding cache...")
    from src.search.similarity_engine import IntegrandGroupSearch

    # manually load CLEmbedder from cache metadata
    cache_info = embedding_cache.get_cache_info('ii-cl-19m_prod')
    checkpoint_path = Path(cache_info['checkpoint_path'])
    config_path = Path(cache_info['config_path']) if cache_info.get('config_path') else None
    logger.info(f"loading E-Gen model from: {checkpoint_path}")
    egen_embedder = CLEmbedder.load(checkpoint_path, config_path=config_path)

    # load cache with pre-loaded embedder
    egen_search_engine = IntegrandGroupSearch.load_cache('ii-cl-19m_prod', egen_embedder, db)
    logger.info(f"loaded E-Gen cache with {egen_search_engine.index.ntotal} vectors")

    # evaluate main model using pre-built cache
    main_metrics, main_results, main_per_family = evaluate_embedder(
        embedder_name="ii-cl-19m",
        embedder=egen_search_engine.embedder,
        db=db,
        queries=queries,
        search_engine=egen_search_engine
    )

    # compare to baselines if requested
    all_embedders = {"ii-cl-19m": (main_metrics, main_results, main_per_family)}

    if args.compare_baselines:
        logger.info("comparing to baseline embedders...")

        # TF-IDF baseline
        try:
            logger.info("loading TF-IDF embedding cache...")
            tfidf_search_engine = IntegrandGroupSearch.load_cache('tfidf', embedder=None, database=db)
            logger.info(f"loaded TF-IDF cache with {tfidf_search_engine.index.ntotal} vectors")

            tfidf_metrics, tfidf_results, tfidf_per_family = evaluate_embedder(
                embedder_name="tf-idf",
                embedder=tfidf_search_engine.embedder,
                db=db,
                queries=queries,
                search_engine=tfidf_search_engine
            )
            all_embedders['tf-idf'] = (tfidf_metrics, tfidf_results, tfidf_per_family)
        except Exception as e:
            logger.warning(f"failed to evaluate TF-IDF baseline: {e}")

        # Sentence-BERT baseline
        try:
            logger.info("loading Sentence-BERT embedding cache...")
            sbert_search_engine = IntegrandGroupSearch.load_cache('sentence_bert', embedder=None, database=db)
            logger.info(f"loaded Sentence-BERT cache with {sbert_search_engine.index.ntotal} vectors")

            sbert_metrics, sbert_results, sbert_per_family = evaluate_embedder(
                embedder_name="sentence-bert",
                embedder=sbert_search_engine.embedder,
                db=db,
                queries=queries,
                search_engine=sbert_search_engine
            )
            all_embedders['sentence-bert'] = (sbert_metrics, sbert_results, sbert_per_family)
        except Exception as e:
            logger.warning(f"failed to evaluate Sentence-BERT baseline: {e}")

    # output results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = paths['data']['root'] / 'evaluation_results' / 'ii-cl-19m_prod'

    output_dir.mkdir(parents=True, exist_ok=True)

    # save metrics JSON
    output_data = {
        'cache': 'ii-cl-19m_prod',
        'model': 'ii-cl-19m_prod',
        'queries_file': str(queries_path),
        'num_queries': len(queries),
        'embedders': {
            name: {
                'metrics': metrics,
                'per_family_metrics': per_family
            }
            for name, (metrics, _, per_family) in all_embedders.items()
        }
    }

    output_path = output_dir / 'retrieval_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"saved retrieval results to {output_path}")

    # generate and save qualitative examples for each embedder
    for embedder_name, (_, results, _) in all_embedders.items():
        examples_text = generate_qualitative_examples(queries, results)
        examples_path = output_dir / f'retrieval_examples_{embedder_name}.txt'
        with open(examples_path, 'w') as f:
            f.write(examples_text)
        logger.info(f"saved qualitative examples for {embedder_name} to {examples_path}")

    # print summary
    logger.info("\n" + "=" * 80)
    logger.info("RETRIEVAL EVALUATION SUMMARY")
    logger.info("=" * 80)

    for embedder_name, (metrics, _, _) in all_embedders.items():
        logger.info(f"\n{embedder_name}:")
        logger.info(f"  Recall@1:  {metrics['recall@1']:.4f}")
        logger.info(f"  Recall@5:  {metrics['recall@5']:.4f}")
        logger.info(f"  Recall@10: {metrics['recall@10']:.4f}")
        logger.info(f"  MRR:       {metrics['mrr']:.4f}")
        logger.info(f"  MAP:       {metrics['map']:.4f}")
        logger.info(f"  NDCG@10:   {metrics['ndcg@10']:.4f}")

    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate retrieval performance on database using pre-built caches')
    parser.add_argument('--compare-baselines', action='store_true', default=True, help='compare to baseline embedders (default: True)')
    parser.add_argument('--no-baselines', dest='compare_baselines', action='store_false', help='skip baseline comparison')
    parser.add_argument('--output-dir', type=str, default=None, help='output directory (default: data/evaluation_results/ii-cl-19m_prod)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main(args)
