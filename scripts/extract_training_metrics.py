"""
extract training metrics from logs and tensorboard

usage:
    python scripts/extract_training_metrics.py --experiment ii-cl-19m_v1.0
    python scripts/extract_training_metrics.py --experiment ii-cl-19m_v1.0 --output evaluation_results/training_metrics.json
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("tensorboard not installed, skipping tensorboard metrics extraction")

from src.utils.paths import get_paths

logger = logging.getLogger(__name__)


def parse_stats_json(stats_path: Path) -> Dict[str, Any]:
    """parse epoch-level metrics from stats.json"""
    with open(stats_path) as f:
        stats = json.load(f)

    metrics = {
        'epochs': [s['epoch'] for s in stats],
        'train_loss': [s['train_loss'] for s in stats],
        'val_loss': [s['val_loss'] for s in stats],
        'learning_rate': [s['learning_rate'] for s in stats],
        'timestamps': [s['timestamp'] for s in stats],
    }

    logger.info(f"extracted {len(stats)} epochs from stats.json")
    return metrics


def parse_tensorboard_events(tb_dir: Path) -> Dict[str, Dict[str, List]]:
    """parse iteration-level metrics from tensorboard event files"""
    if not TENSORBOARD_AVAILABLE:
        logger.warning("tensorboard not available, returning empty metrics")
        return {}

    # find all event files
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        logger.warning(f"no tensorboard event files found in {tb_dir}")
        return {}

    logger.info(f"found {len(event_files)} tensorboard event files")

    # parse all events
    all_metrics = {}
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()

            # extract scalar tags
            for tag in ea.Tags()['scalars']:
                if tag not in all_metrics:
                    all_metrics[tag] = {'steps': [], 'values': [], 'timestamps': []}

                events = ea.Scalars(tag)
                for event in events:
                    all_metrics[tag]['steps'].append(event.step)
                    all_metrics[tag]['values'].append(event.value)
                    all_metrics[tag]['timestamps'].append(event.wall_time)

        except Exception as e:
            logger.warning(f"failed to parse {event_file.name}: {e}")
            continue

    # sort by step
    for tag in all_metrics:
        sorted_indices = np.argsort(all_metrics[tag]['steps'])
        all_metrics[tag]['steps'] = [all_metrics[tag]['steps'][i] for i in sorted_indices]
        all_metrics[tag]['values'] = [all_metrics[tag]['values'][i] for i in sorted_indices]
        all_metrics[tag]['timestamps'] = [all_metrics[tag]['timestamps'][i] for i in sorted_indices]

    logger.info(f"extracted {len(all_metrics)} metric types from tensorboard")
    for tag in sorted(all_metrics.keys()):
        logger.info(f"  {tag}: {len(all_metrics[tag]['steps'])} data points")

    return all_metrics


def plot_learning_curves(epoch_metrics: Dict, tb_metrics: Dict, output_dir: Path):
    """generate learning curve plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # epoch-level plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss curves
    ax1.plot(epoch_metrics['epochs'], epoch_metrics['train_loss'], 'o-', label='Train Loss', linewidth=2)
    ax1.plot(epoch_metrics['epochs'], epoch_metrics['val_loss'], 's-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress (Epoch-Level)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # learning rate
    ax2.plot(epoch_metrics['epochs'], epoch_metrics['learning_rate'], 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'learning_curves_epoch.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"saved epoch-level learning curves to {output_path}")
    plt.close()

    # iteration-level plot (if available)
    if 'train/loss' in tb_metrics and len(tb_metrics['train/loss']['steps']) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        train_steps = tb_metrics['train/loss']['steps']
        train_values = tb_metrics['train/loss']['values']

        # smooth with moving average
        window_size = max(1, len(train_values) // 200)
        if window_size > 1:
            smoothed = np.convolve(train_values, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = train_steps[:len(smoothed)]
            ax.plot(smoothed_steps, smoothed, '-', label='Train Loss (smoothed)', linewidth=2, alpha=0.8)

        # raw data with transparency
        ax.plot(train_steps, train_values, '-', alpha=0.2, linewidth=0.5, color='blue', label='Train Loss (raw)')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Over Iterations', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'learning_curves_iteration.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"saved iteration-level learning curves to {output_path}")
        plt.close()


def extract_summary_statistics(epoch_metrics: Dict, tb_metrics: Dict) -> Dict[str, Any]:
    """compute summary statistics from training metrics"""
    summary = {
        'n_epochs': len(epoch_metrics['epochs']),
        'final_train_loss': epoch_metrics['train_loss'][-1],
        'final_val_loss': epoch_metrics['val_loss'][-1],
        'best_val_loss': min(epoch_metrics['val_loss']),
        'best_epoch': int(np.argmin(epoch_metrics['val_loss']) + 1),
        'final_lr': epoch_metrics['learning_rate'][-1],
    }

    # compute training convergence metrics
    train_losses = np.array(epoch_metrics['train_loss'])
    val_losses = np.array(epoch_metrics['val_loss'])

    # relative improvement
    summary['train_loss_improvement'] = float((train_losses[0] - train_losses[-1]) / train_losses[0])
    summary['val_loss_improvement'] = float((val_losses[0] - val_losses[-1]) / val_losses[0])

    # overfitting check
    summary['val_train_gap'] = float(val_losses[-1] - train_losses[-1])
    summary['overfit_ratio'] = float(val_losses[-1] / train_losses[-1])

    # iteration-level stats
    if 'train/loss' in tb_metrics:
        summary['total_iterations'] = len(tb_metrics['train/loss']['steps'])
        summary['final_iteration'] = int(tb_metrics['train/loss']['steps'][-1])
    else:
        summary['total_iterations'] = None
        summary['final_iteration'] = None

    # check for evaluation metrics
    eval_tags = [tag for tag in tb_metrics.keys() if tag.startswith('eval/')]
    if eval_tags:
        summary['evaluation_metrics_available'] = True
        summary['evaluation_tags'] = eval_tags
        logger.info(f"found {len(eval_tags)} evaluation metric types in tensorboard")
    else:
        summary['evaluation_metrics_available'] = False
        summary['evaluation_tags'] = []
        logger.warning("no evaluation metrics found in tensorboard (tags starting with 'eval/')")

    return summary


def main(args):
    paths = get_paths()

    # find experiment directory
    runs_dir = paths['models']['runs']
    experiment_dir = runs_dir / args.experiment

    if not experiment_dir.exists():
        raise FileNotFoundError(f"experiment directory not found: {experiment_dir}")

    logger.info(f"extracting metrics from experiment: {args.experiment}")
    logger.info(f"experiment directory: {experiment_dir}")

    # parse stats.json
    stats_path = experiment_dir / 'stats.json'
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found: {stats_path}")

    epoch_metrics = parse_stats_json(stats_path)

    # parse tensorboard events
    tb_dir = experiment_dir / 'tensorboard'
    tb_metrics = {}
    if tb_dir.exists():
        tb_metrics = parse_tensorboard_events(tb_dir)
    else:
        logger.warning(f"tensorboard directory not found: {tb_dir}")

    # compute summary statistics
    summary = extract_summary_statistics(epoch_metrics, tb_metrics)

    # log summary
    logger.info("=" * 60)
    logger.info("training summary:")
    logger.info(f"  epochs trained: {summary['n_epochs']}")
    logger.info(f"  total iterations: {summary['total_iterations']}")
    logger.info(f"  final train loss: {summary['final_train_loss']:.6f}")
    logger.info(f"  final val loss: {summary['final_val_loss']:.6f}")
    logger.info(f"  best val loss: {summary['best_val_loss']:.6f} (epoch {summary['best_epoch']})")
    logger.info(f"  train loss improvement: {summary['train_loss_improvement']*100:.1f}%")
    logger.info(f"  val loss improvement: {summary['val_loss_improvement']*100:.1f}%")
    logger.info(f"  val/train gap: {summary['val_train_gap']:.6f} (ratio: {summary['overfit_ratio']:.3f})")
    logger.info(f"  evaluation metrics available: {summary['evaluation_metrics_available']}")
    logger.info("=" * 60)

    # generate plots
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = paths['project_root'] / 'evaluation_results' / args.experiment

    figures_dir = output_dir / 'figures'
    plot_learning_curves(epoch_metrics, tb_metrics, figures_dir)

    # export full metrics
    output_data = {
        'experiment': args.experiment,
        'summary': summary,
        'epoch_metrics': epoch_metrics,
        'tensorboard_metrics': {
            tag: {
                'steps': tb_metrics[tag]['steps'],
                'values': tb_metrics[tag]['values'],
            }
            for tag in tb_metrics.keys()
        } if tb_metrics else {},
    }

    output_path = output_dir / 'training_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"saved training metrics to {output_path}")
    logger.info(f"saved figures to {figures_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract training metrics from logs and tensorboard')
    parser.add_argument('--experiment', type=str, required=True, help='experiment name (e.g., ii-cl-19m_v1.0)')
    parser.add_argument('--output-dir', type=str, default=None, help='output directory (default: evaluation_results/<experiment>)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    main(args)
