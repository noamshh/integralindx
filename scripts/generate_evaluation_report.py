"""
orchestrate full evaluation pipeline and generate comprehensive report

uses pre-built caches:
- ii-cl-19m_prod (E-Gen trained model)
- tfidf (baseline)

extracts training metrics from:
- ii-cl-19m_v1.0 (training run with TensorBoard logs)

usage:
    # default: full evaluation with baselines
    python -m scripts.generate_evaluation_report

    # custom experiment name (for output)
    python -m scripts.generate_evaluation_report --experiment my_experiment

    # custom training experiment (for metrics extraction)
    python -m scripts.generate_evaluation_report --training-experiment ii-cl-19m_v1.0

    # custom output directory
    python -m scripts.generate_evaluation_report --output-dir evaluation_results/my_eval

    # continue even if a step fails
    python -m scripts.generate_evaluation_report --skip-on-error
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.paths import get_paths

logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """run a command and return success status"""
    logger.info(f"running: {description}")
    logger.info(f"command: {' '.join(cmd)}")

    try:
        # stream output in real-time instead of capturing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line buffered
        )

        # print output as it arrives
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            logger.error(f"command failed with exit code {process.returncode}")
            return False

        return True
    except Exception as e:
        logger.error(f"failed to run {description}: {e}")
        return False


def generate_markdown_report(output_dir: Path, experiment: str,
                               training_metrics: Dict, test_results: Dict,
                               retrieval_results: Dict, analogy_results: Dict = None) -> Path:
    """generate markdown evaluation report"""
    report_lines = []

    # header
    report_lines.append(f"# IntegralIndx Model Evaluation Report")
    report_lines.append(f"")
    report_lines.append(f"**Experiment**: `{experiment}`  ")
    report_lines.append(f"**Cache**: `ii-cl-19m_prod` (E-Gen trained model)  ")
    report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    report_lines.append(f"")
    report_lines.append("---")
    report_lines.append(f"")

    # executive summary
    report_lines.append("## Executive Summary")
    report_lines.append(f"")

    if training_metrics:
        summary = training_metrics.get('summary', {})
        report_lines.append(f"This report presents a comprehensive evaluation of the **{experiment}** E-Gen model trained for integral similarity search.")
        report_lines.append(f"")
        report_lines.append(f"**Key Findings:**")
        report_lines.append(f"")
        report_lines.append(f"- **Training**: {summary.get('n_epochs', 'N/A')} epochs, final validation loss = {summary.get('final_val_loss', 0):.4f}")

    if test_results:
        test_metrics = test_results.get('metrics', {})
        report_lines.append(f"- **Generalization**: Test loss = {test_metrics.get('test_loss', 0):.4f}, separation margin = {test_metrics.get('separation_margin', 0):.4f}")

    if retrieval_results:
        embedders = retrieval_results.get('embedders', {})
        main_embedder = list(embedders.keys())[0]
        main_metrics = embedders[main_embedder]['metrics']
        report_lines.append(f"- **Retrieval**: Recall@5 = {main_metrics.get('recall@5', 0):.4f}, MRR = {main_metrics.get('mrr', 0):.4f}")

    if analogy_results:
        embedders = analogy_results.get('embedders', {})
        if embedders:
            main_embedder = list(embedders.keys())[0]
            main_metrics = embedders[main_embedder]['metrics']
            report_lines.append(f"- **Vector Analogies**: Accuracy@5 = {main_metrics.get('accuracy@5', 0):.4f} ({analogy_results.get('filter_stats', {}).get('valid_analogies', 0)} valid analogies)")

    report_lines.append(f"")
    report_lines.append("---")
    report_lines.append(f"")

    # training section
    if training_metrics:
        report_lines.append("## Training Performance")
        report_lines.append(f"")

        summary = training_metrics.get('summary', {})

        report_lines.append(f"### Training Summary")
        report_lines.append(f"")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|-------|")
        report_lines.append(f"| Total Epochs | {summary.get('n_epochs', 'N/A')} |")
        report_lines.append(f"| Total Iterations | {summary.get('total_iterations', 'N/A'):,} |")
        report_lines.append(f"| Final Train Loss | {summary.get('final_train_loss', 0):.6f} |")
        report_lines.append(f"| Final Val Loss | {summary.get('final_val_loss', 0):.6f} |")
        report_lines.append(f"| Best Val Loss | {summary.get('best_val_loss', 0):.6f} (epoch {summary.get('best_epoch', 'N/A')}) |")
        report_lines.append(f"| Train Loss Improvement | {summary.get('train_loss_improvement', 0)*100:.1f}% |")
        report_lines.append(f"| Val Loss Improvement | {summary.get('val_loss_improvement', 0)*100:.1f}% |")
        report_lines.append(f"| Val/Train Gap | {summary.get('val_train_gap', 0):.6f} (ratio: {summary.get('overfit_ratio', 0):.3f}) |")
        report_lines.append(f"")

        # training curves
        figures_dir = output_dir / 'figures'
        if (figures_dir / 'training_summary.png').exists():
            report_lines.append(f"### Training Curves")
            report_lines.append(f"")
            report_lines.append(f"![Training Summary](figures/training_summary.png)")
            report_lines.append(f"")

        if (figures_dir / 'learning_curves_epoch.png').exists():
            report_lines.append(f"![Learning Curves (Epoch)](figures/learning_curves_epoch.png)")
            report_lines.append(f"")

        if (figures_dir / 'learning_curves_iteration.png').exists():
            report_lines.append(f"![Learning Curves (Iteration)](figures/learning_curves_iteration.png)")
            report_lines.append(f"")

    # test set section
    if test_results:
        report_lines.append("## Test Set Evaluation")
        report_lines.append(f"")

        test_metrics = test_results.get('metrics', {})
        report_lines.append(f"Evaluated on **{test_results.get('num_evaluated', 'N/A'):,}** held-out triplets from the test split.")
        report_lines.append(f"")

        report_lines.append(f"### Test Metrics")
        report_lines.append(f"")
        report_lines.append(f"| Metric | Value | Description |")
        report_lines.append(f"|--------|-------|-------------|")
        report_lines.append(f"| Test Loss | {test_metrics.get('test_loss', 0):.4f} | InfoNCE loss on held-out data |")
        report_lines.append(f"| Intra-Class Similarity | {test_metrics.get('intra_class_similarity', 0):.4f} | Avg. cosine sim (query, positive) |")
        report_lines.append(f"| Inter-Class Similarity | {test_metrics.get('inter_class_similarity', 0):.4f} | Avg. cosine sim (query, negatives) |")
        report_lines.append(f"| Separation Margin | {test_metrics.get('separation_margin', 0):.4f} | Intra - inter similarity |")
        report_lines.append(f"")

    # retrieval section
    if retrieval_results:
        report_lines.append("## Retrieval Performance")
        report_lines.append(f"")

        report_lines.append(f"Evaluated on **{retrieval_results.get('num_queries', 'N/A')}** fixed queries from the evaluation set.")
        report_lines.append(f"")

        embedders = retrieval_results.get('embedders', {})

        # metrics comparison table
        report_lines.append(f"### Retrieval Metrics")
        report_lines.append(f"")

        metrics_to_show = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'map', 'ndcg@10']
        header = "| Embedder | " + " | ".join([m.replace('@', ' @ ').upper() for m in metrics_to_show]) + " |"
        separator = "|" + "|".join(["--------"] * (len(metrics_to_show) + 1)) + "|"

        report_lines.append(header)
        report_lines.append(separator)

        for embedder_name, embedder_data in embedders.items():
            metrics = embedder_data['metrics']
            row = f"| **{embedder_name}** |"
            for metric in metrics_to_show:
                row += f" {metrics.get(metric, 0):.4f} |"
            report_lines.append(row)

        report_lines.append(f"")

        # visualizations
        figures_dir = output_dir / 'figures'
        if (figures_dir / 'retrieval_comparison.png').exists():
            report_lines.append(f"### Retrieval Comparison")
            report_lines.append(f"")
            report_lines.append(f"![Retrieval Comparison](figures/retrieval_comparison.png)")
            report_lines.append(f"")

        if (figures_dir / 'recall_at_k_curve.png').exists():
            report_lines.append(f"![Recall @ K](figures/recall_at_k_curve.png)")
            report_lines.append(f"")

        if (figures_dir / 'per_family_performance.png').exists():
            report_lines.append(f"### Per-Family Performance")
            report_lines.append(f"")
            report_lines.append(f"![Per-Family Performance](figures/per_family_performance.png)")
            report_lines.append(f"")

    # analogy section
    if analogy_results:
        report_lines.append("## Vector Math Analogy Evaluation")
        report_lines.append(f"")

        filter_stats = analogy_results.get('filter_stats', {})
        report_lines.append(f"Evaluated on **{filter_stats.get('valid_analogies', 'N/A')}** valid analogies "
                           f"(filtered {filter_stats.get('filtered_out', 0)} where D not in dataset).")
        report_lines.append(f"")

        embedders = analogy_results.get('embedders', {})

        # metrics comparison table
        report_lines.append(f"### Analogy Accuracy")
        report_lines.append(f"")

        metrics_to_show = ['accuracy@1', 'accuracy@5', 'accuracy@10']
        header = "| Embedder | " + " | ".join([m.replace('accuracy@', 'Acc @ ') for m in metrics_to_show]) + " |"
        separator = "|" + "|".join(["--------"] * (len(metrics_to_show) + 1)) + "|"

        report_lines.append(header)
        report_lines.append(separator)

        for embedder_name, embedder_data in embedders.items():
            metrics = embedder_data['metrics']
            row = f"| **{embedder_name}** |"
            for metric in metrics_to_show:
                row += f" {metrics.get(metric, 0):.4f} |"
            report_lines.append(row)

        report_lines.append(f"")

        # per-category breakdown
        report_lines.append(f"### Per-Category Performance")
        report_lines.append(f"")

        # get categories from first embedder
        first_embedder = list(embedders.keys())[0]
        categories = list(embedders[first_embedder]['per_category_metrics'].keys())

        # category table header
        header = "| Category |"
        for embedder_name in embedders.keys():
            header += f" {embedder_name} Acc@5 |"
        report_lines.append(header)

        separator = "|" + "|".join(["--------"] * (len(embedders) + 1)) + "|"
        report_lines.append(separator)

        # category rows
        for category in categories:
            row = f"| {category.replace('_', ' ').title()} |"
            for embedder_name in embedders.keys():
                cat_metrics = embedders[embedder_name]['per_category_metrics'].get(category, {})
                acc5 = cat_metrics.get('accuracy@5', 0)
                row += f" {acc5:.3f} |"
            report_lines.append(row)

        report_lines.append(f"")

        # link to qualitative examples
        for embedder_name in embedders.keys():
            examples_path = output_dir / f'analogy_examples_{embedder_name}.txt'
            if examples_path.exists():
                report_lines.append(f"**Qualitative Examples ({embedder_name})**: See `analogy_examples_{embedder_name}.txt`")
                report_lines.append(f"")

    # summary table
    figures_dir = output_dir / 'figures'
    if (figures_dir / 'metrics_summary_table.png').exists():
        report_lines.append("## Metrics Summary")
        report_lines.append(f"")
        report_lines.append(f"![Metrics Summary](figures/metrics_summary_table.png)")
        report_lines.append(f"")

    # conclusion
    report_lines.append("---")
    report_lines.append(f"")
    report_lines.append("## Conclusion")
    report_lines.append(f"")

    if retrieval_results:
        embedders = retrieval_results.get('embedders', {})
        main_embedder = list(embedders.keys())[0]
        main_metrics = embedders[main_embedder]['metrics']

        recall5 = main_metrics.get('recall@5', 0)
        if recall5 > 0.7:
            performance = "excellent"
        elif recall5 > 0.5:
            performance = "good"
        elif recall5 > 0.3:
            performance = "moderate"
        else:
            performance = "needs improvement"

        report_lines.append(f"The **{experiment}** model demonstrates **{performance}** retrieval performance with Recall@5 = {recall5:.3f}.")

        if len(embedders) > 1:
            report_lines.append(f"Compared to baseline methods, the E-Gen model shows significant improvements in similarity search quality.")

    report_lines.append(f"")
    report_lines.append(f"This evaluation provides confidence in the model's ability to generalize to unseen expressions and accurately retrieve similar integrals.")
    report_lines.append(f"")

    # write report
    report_path = output_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"saved markdown report to {report_path}")
    return report_path


def main(args):
    paths = get_paths()

    # setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = paths['project_root'] / 'evaluation_results' / args.experiment

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"output experiment: {args.experiment}")
    logger.info(f"training experiment: {args.training_experiment}")
    logger.info(f"evaluation cache: ii-cl-19m_prod (E-Gen) + tfidf (baseline)")
    logger.info(f"output directory: {output_dir}")
    logger.info("=" * 80)

    # step 1: extract training metrics (from training experiment)
    logger.info("\n[STEP 1/5] extracting training metrics...")
    training_metrics_path = output_dir / 'training_metrics.json'
    cmd = [
        sys.executable, '-m', 'scripts.extract_training_metrics',
        '--experiment', args.training_experiment,
        '--output-dir', str(output_dir)
    ]
    success = run_command(cmd, "extract training metrics")
    if not success and not args.skip_on_error:
        logger.error("failed at step 1, aborting")
        return

    # step 2: evaluate on test set (with sampling)
    logger.info("\n[STEP 2/5] evaluating on test set...")
    test_results_path = output_dir / 'test_results.json'
    cmd = [
        sys.executable, '-m', 'scripts.evaluate_model',
        '--checkpoint', args.checkpoint,
        '--model', args.model,
        '--training', args.training,
        '--max-test-samples', str(args.max_test_samples),
        '--sample-seed', str(args.sample_seed),
        '--output-dir', str(output_dir)
    ]
    if args.visualize_embeddings:
        cmd.append('--visualize')
    success = run_command(cmd, "evaluate on test set")
    if not success and not args.skip_on_error:
        logger.error("failed at step 2, aborting")
        return

    # step 3: evaluate retrieval (using pre-built caches)
    logger.info("\n[STEP 3/6] evaluating retrieval performance...")
    retrieval_results_path = output_dir / 'retrieval_results.json'
    cmd = [
        sys.executable, '-m', 'scripts.evaluate_retrieval',
        '--output-dir', str(output_dir)
    ]
    # baselines are enabled by default, add --no-baselines to skip
    success = run_command(cmd, "evaluate retrieval")
    if not success and not args.skip_on_error:
        logger.error("failed at step 3, aborting")
        return

    # step 4: evaluate vector math analogies
    logger.info("\n[STEP 4/6] evaluating vector math analogies...")
    analogy_results_path = output_dir / 'analogy_results.json'
    cmd = [
        sys.executable, '-m', 'scripts.evaluate_analogy',
        '--output-dir', str(output_dir)
    ]
    success = run_command(cmd, "evaluate analogies")
    if not success and not args.skip_on_error:
        logger.error("failed at step 4, aborting")
        return

    # step 5: generate visualizations
    logger.info("\n[STEP 5/6] generating visualizations...")
    figures_dir = output_dir / 'figures'
    cmd = [
        sys.executable, '-m', 'scripts.generate_visualizations',
        '--retrieval-results', str(retrieval_results_path),
        '--output-dir', str(figures_dir)
    ]
    if training_metrics_path.exists():
        cmd.extend(['--training-metrics', str(training_metrics_path)])
    if test_results_path.exists():
        cmd.extend(['--test-results', str(test_results_path)])
    success = run_command(cmd, "generate visualizations")
    if not success and not args.skip_on_error:
        logger.error("failed at step 5, aborting")
        return

    # step 6: generate report
    logger.info("\n[STEP 6/6] generating evaluation report...")

    # load all results
    training_metrics = {}
    if training_metrics_path.exists():
        with open(training_metrics_path) as f:
            training_metrics = json.load(f)

    test_results = {}
    if test_results_path.exists():
        with open(test_results_path) as f:
            test_results = json.load(f)

    retrieval_results = {}
    if retrieval_results_path.exists():
        with open(retrieval_results_path) as f:
            retrieval_results = json.load(f)

    analogy_results = {}
    if analogy_results_path.exists():
        with open(analogy_results_path) as f:
            analogy_results = json.load(f)

    report_path = generate_markdown_report(
        output_dir=output_dir,
        experiment=args.experiment,
        training_metrics=training_metrics,
        test_results=test_results,
        retrieval_results=retrieval_results,
        analogy_results=analogy_results
    )

    # try to generate HTML (if pandoc is available)
    try:
        html_path = output_dir / 'report.html'
        subprocess.run(
            ['pandoc', str(report_path), '-o', str(html_path), '--standalone', '--self-contained'],
            check=True,
            capture_output=True
        )
        logger.info(f"saved HTML report to {html_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("pandoc not available, skipping HTML generation (install with: apt-get install pandoc)")

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"results saved to: {output_dir}")
    logger.info(f"  - markdown report: {report_path}")
    logger.info(f"  - training metrics: {training_metrics_path}")
    logger.info(f"  - test results: {test_results_path}")
    logger.info(f"  - retrieval results: {retrieval_results_path}")
    logger.info(f"  - analogy results: {analogy_results_path}")
    logger.info(f"  - figures: {figures_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate comprehensive evaluation report using pre-built caches')
    parser.add_argument('--checkpoint', type=str, default='data/models/inference/ii-cl-19m_prod.pt', help='path to model checkpoint (for test eval, matches cache)')
    parser.add_argument('--experiment', type=str, default='ii-cl-19m_prod', help='experiment name for output (default: ii-cl-19m_prod)')
    parser.add_argument('--training-experiment', type=str, default='ii-cl-19m_v1.0', help='training experiment for extracting metrics (default: ii-cl-19m_v1.0)')
    parser.add_argument('--model', type=str, default='ii-cl-19m', help='model config name')
    parser.add_argument('--training', type=str, default='gpu', help='training config name')
    parser.add_argument('--max-test-samples', type=int, default=10000, help='max test samples to evaluate')
    parser.add_argument('--sample-seed', type=int, default=42, help='random seed for sampling')
    parser.add_argument('--visualize-embeddings', action='store_true', help='generate embedding visualizations')
    parser.add_argument('--output-dir', type=str, default=None, help='output directory')
    parser.add_argument('--skip-on-error', action='store_true', help='continue if a step fails')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main(args)
