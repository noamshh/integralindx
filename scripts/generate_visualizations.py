"""
generate comprehensive visualizations for evaluation report

usage:
    python scripts/generate_visualizations.py --retrieval-results evaluation_results/retrieval_results.json --output-dir evaluation_results/figures
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_retrieval_comparison(embedders_data: Dict[str, Dict], output_dir: Path):
    """generate bar chart comparing embedders across metrics"""
    metrics_to_plot = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'map', 'ndcg@10']
    embedder_names = list(embedders_data.keys())

    # prepare data
    data = {metric: [] for metric in metrics_to_plot}
    for embedder_name in embedder_names:
        metrics = embedders_data[embedder_name]['metrics']
        for metric in metrics_to_plot:
            data[metric].append(metrics.get(metric, 0.0))

    # create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        x_pos = np.arange(len(embedder_names))
        bars = ax.bar(x_pos, data[metric], alpha=0.8, color=['#2ecc71', '#3498db', '#e74c3c'][:len(embedder_names)])

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.upper().replace('@', ' @ '), fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in embedder_names], rotation=15, ha='right')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, data[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Retrieval Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / 'retrieval_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"saved retrieval comparison to {output_path}")
    plt.close()


def plot_per_family_performance(embedders_data: Dict[str, Dict], output_dir: Path, top_n: int = 15):
    """generate heatmap of per-family recall@5"""
    # extract per-family metrics for main embedder (first one)
    main_embedder = list(embedders_data.keys())[0]
    per_family = embedders_data[main_embedder].get('per_family_metrics', {})

    if not per_family:
        logger.warning("no per-family metrics available, skipping family performance plot")
        return

    # sort families by count
    families_sorted = sorted(per_family.items(), key=lambda x: x[1]['count'], reverse=True)[:top_n]

    # prepare data
    family_names = [f[0] for f in families_sorted]
    metrics_names = ['recall@1', 'recall@5', 'recall@10', 'mrr']

    data_matrix = []
    for family_name in family_names:
        row = [per_family[family_name][metric] for metric in metrics_names]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        xticklabels=[m.replace('@', ' @ ').upper() for m in metrics_names],
        yticklabels=[f[:25] for f in family_names],
        cbar_kws={'label': 'Score'},
        vmin=0,
        vmax=1.0,
        ax=ax
    )

    ax.set_title(f'Per-Family Performance (Top {top_n} Families by Query Count)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Family', fontsize=12)

    plt.tight_layout()

    output_path = output_dir / 'per_family_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"saved per-family performance to {output_path}")
    plt.close()


def plot_recall_at_k_curve(embedders_data: Dict[str, Dict], output_dir: Path, max_k: int = 20):
    """plot recall@k for different k values"""
    # we need to recompute this with varying k, so we'll just plot what we have
    embedder_names = list(embedders_data.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = [1, 5, 10, 20]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    for idx, embedder_name in enumerate(embedder_names):
        metrics = embedders_data[embedder_name]['metrics']
        recall_values = [metrics.get(f'recall@{k}', 0.0) for k in k_values]

        ax.plot(k_values, recall_values, 'o-', label=embedder_name.replace('_', ' ').title(),
               linewidth=2, markersize=8, color=colors[idx % len(colors)])

    ax.set_xlabel('k (Number of Retrieved Results)', fontsize=12)
    ax.set_ylabel('Recall @ k', fontsize=12)
    ax.set_title('Recall vs. Number of Retrieved Results', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, max(k_values) + 1])

    plt.tight_layout()

    output_path = output_dir / 'recall_at_k_curve.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"saved recall@k curve to {output_path}")
    plt.close()


def plot_training_summary(training_metrics_path: Path, output_dir: Path):
    """generate training summary plots"""
    if not training_metrics_path or not training_metrics_path.exists():
        logger.warning(f"training metrics not found at {training_metrics_path}, skipping training plots")
        return

    with open(training_metrics_path) as f:
        training_data = json.load(f)

    epoch_metrics = training_data.get('epoch_metrics', {})
    if not epoch_metrics:
        logger.warning("no epoch metrics in training data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss curves
    epochs = epoch_metrics['epochs']
    train_loss = epoch_metrics['train_loss']
    val_loss = epoch_metrics['val_loss']

    ax1.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2.5, markersize=8)
    ax1.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # loss improvement
    train_improvement = [(train_loss[0] - l) / train_loss[0] * 100 for l in train_loss]
    val_improvement = [(val_loss[0] - l) / val_loss[0] * 100 for l in val_loss]

    ax2.plot(epochs, train_improvement, 'o-', label='Train Loss Improvement', linewidth=2.5, markersize=8)
    ax2.plot(epochs, val_improvement, 's-', label='Val Loss Improvement', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Loss Improvement Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    output_path = output_dir / 'training_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"saved training summary to {output_path}")
    plt.close()


def plot_metrics_summary_table(embedders_data: Dict[str, Dict], training_summary: Dict,
                                 test_results: Dict, output_dir: Path):
    """generate a visual summary table of all key metrics"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # prepare table data
    table_data = []

    # header
    table_data.append(['Metric', 'Value', 'Description'])

    # training metrics
    table_data.append(['=== TRAINING ===', '', ''])
    table_data.append(['Epochs', f"{training_summary.get('n_epochs', 'N/A')}", 'Total epochs trained'])
    table_data.append(['Final Train Loss', f"{training_summary.get('final_train_loss', 0):.4f}", 'Training loss at end'])
    table_data.append(['Final Val Loss', f"{training_summary.get('final_val_loss', 0):.4f}", 'Validation loss at end'])
    table_data.append(['Best Val Loss', f"{training_summary.get('best_val_loss', 0):.4f}", f"Best validation (epoch {training_summary.get('best_epoch', 'N/A')})"])

    # test set metrics
    if test_results:
        test_metrics = test_results.get('metrics', {})
        table_data.append(['=== TEST SET ===', '', ''])
        table_data.append(['Test Loss', f"{test_metrics.get('test_loss', 0):.4f}", 'Held-out test triplets'])
        table_data.append(['Intra-Class Sim', f"{test_metrics.get('intra_class_similarity', 0):.4f}", 'Query-positive similarity'])
        table_data.append(['Inter-Class Sim', f"{test_metrics.get('inter_class_similarity', 0):.4f}", 'Query-negative similarity'])
        table_data.append(['Separation Margin', f"{test_metrics.get('separation_margin', 0):.4f}", 'Intra - inter similarity'])

    # retrieval metrics (main embedder)
    main_embedder = list(embedders_data.keys())[0]
    retrieval_metrics = embedders_data[main_embedder]['metrics']
    table_data.append(['=== RETRIEVAL ===', '', ''])
    table_data.append(['Recall @ 1', f"{retrieval_metrics.get('recall@1', 0):.4f}", 'Top-1 accuracy'])
    table_data.append(['Recall @ 5', f"{retrieval_metrics.get('recall@5', 0):.4f}", 'Top-5 accuracy'])
    table_data.append(['Recall @ 10', f"{retrieval_metrics.get('recall@10', 0):.4f}", 'Top-10 accuracy'])
    table_data.append(['MRR', f"{retrieval_metrics.get('mrr', 0):.4f}", 'Mean reciprocal rank'])
    table_data.append(['MAP', f"{retrieval_metrics.get('map', 0):.4f}", 'Mean average precision'])
    table_data.append(['NDCG @ 10', f"{retrieval_metrics.get('ndcg@10', 0):.4f}", 'Normalized DCG'])

    # create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # style section headers
    for row_idx, row in enumerate(table_data):
        if row[0].startswith('==='):
            for col_idx in range(3):
                table[(row_idx, col_idx)].set_facecolor('#ecf0f1')
                table[(row_idx, col_idx)].set_text_props(weight='bold')

    ax.set_title('Evaluation Metrics Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = output_dir / 'metrics_summary_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"saved metrics summary table to {output_path}")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load retrieval results
    retrieval_path = Path(args.retrieval_results)
    if not retrieval_path.exists():
        raise FileNotFoundError(f"retrieval results not found: {retrieval_path}")

    with open(retrieval_path) as f:
        retrieval_data = json.load(f)

    embedders_data = retrieval_data['embedders']

    logger.info(f"generating visualizations for {len(embedders_data)} embedders")

    # generate plots
    plot_retrieval_comparison(embedders_data, output_dir)
    plot_per_family_performance(embedders_data, output_dir)
    plot_recall_at_k_curve(embedders_data, output_dir)

    # training plots (if available)
    if args.training_metrics:
        training_path = Path(args.training_metrics)
        plot_training_summary(training_path, output_dir)
    else:
        logger.info("no training metrics provided, skipping training plots")

    # summary table
    training_summary = {}
    if args.training_metrics:
        with open(args.training_metrics) as f:
            training_summary = json.load(f).get('summary', {})

    test_results = {}
    if args.test_results:
        with open(args.test_results) as f:
            test_results = json.load(f)

    plot_metrics_summary_table(embedders_data, training_summary, test_results, output_dir)

    logger.info(f"all visualizations saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate comprehensive visualizations for evaluation')
    parser.add_argument('--retrieval-results', type=str, required=True, help='path to retrieval_results.json')
    parser.add_argument('--training-metrics', type=str, default=None, help='path to training_metrics.json (optional)')
    parser.add_argument('--test-results', type=str, default=None, help='path to test_results.json (optional)')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory for figures')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main(args)
