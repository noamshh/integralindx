#!/usr/bin/env python3
"""
Extract training/validation loss and learning rate from TensorBoard logs
and generate a professional training curves figure.

Usage:
    python scripts/plot_training_curves.py
"""
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

def extract_tensorboard_scalars(log_dir, scalar_name):
    """
    extract scalar values from TensorBoard event files

    args:
        log_dir: path to TensorBoard log directory
        scalar_name: name of scalar to extract (e.g., 'train/loss')

    returns:
        list of (step, value) tuples
    """
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    if scalar_name not in ea.Tags()['scalars']:
        print(f"Warning: {scalar_name} not found in event files")
        return []

    events = ea.Scalars(scalar_name)
    return [(e.step, e.value) for e in events]

def exponential_moving_average(values, alpha=0.6):
    """
    apply exponential moving average smoothing (TensorBoard-style)

    args:
        values: list of values to smooth
        alpha: smoothing factor (0-1, lower = more smoothing)

    returns:
        smoothed values
    """
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        smoothed_val = last * alpha + (1 - alpha) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_training_curves(tb_log_dir, output_path):
    """
    create training curves figure with train/val loss and learning rate

    args:
        tb_log_dir: path to TensorBoard log directory
        output_path: where to save the figure (SVG format)
    """
    print(f"Extracting data from TensorBoard logs: {tb_log_dir}")

    # extract data
    train_loss = extract_tensorboard_scalars(tb_log_dir, 'train/loss')
    val_loss = extract_tensorboard_scalars(tb_log_dir, 'val/loss')
    lr = extract_tensorboard_scalars(tb_log_dir, 'train/learning_rate')

    print(f"  Train loss points: {len(train_loss)}")
    print(f"  Val loss points: {len(val_loss)}")
    print(f"  Learning rate points: {len(lr)}")

    if not train_loss:
        print("Error: No training data found!")
        return

    # create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()

    # plot losses on primary axis
    steps_train, loss_train = zip(*train_loss)

    # apply exponential moving average smoothing to train loss
    # alpha=0.9 gives very smooth curve (lower alpha = more smoothing)
    loss_train_smoothed = exponential_moving_average(list(loss_train), alpha=0.95)

    # plot raw train loss (transparent)
    ax1.plot(steps_train, loss_train, color='#1f77b4', linewidth=0.8, alpha=0.25)

    # plot smoothed train loss (full opacity)
    ax1.plot(steps_train, loss_train_smoothed, label='Train Loss (smoothed)',
             color='#1f77b4', linewidth=2.0, alpha=1.0)

    if val_loss:
        steps_val, loss_val = zip(*val_loss)
        ax1.plot(steps_val, loss_val, label='Val Loss', color='#ff7f0e', linewidth=2.0,
                 linestyle='--', alpha=0.95, marker='o', markersize=4, markevery=1)

    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (InfoNCE)', fontsize=13, fontweight='bold', color='#1f77b4')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)

    # plot learning rate on secondary axis
    if lr:
        steps_lr, lr_vals = zip(*lr)
        ax2.plot(steps_lr, lr_vals, label='Learning Rate', color='#2ca02c',
                 linewidth=1.5, linestyle=':', alpha=0.85)
        ax2.set_ylabel('Learning Rate', fontsize=13, fontweight='bold', color='#2ca02c')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='#2ca02c', labelsize=11)
        ax2.spines['right'].set_color('#2ca02c')
        ax2.spines['right'].set_linewidth(1.5)

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11, framealpha=0.95)

    # title
    plt.title('Training and Validation Loss Curves (ii-cl-19m v1.0)',
              fontsize=15, fontweight='bold', pad=15)

    # styling
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color('#1f77b4')
    ax1.spines['left'].set_linewidth(1.5)

    plt.tight_layout()

    # save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format='svg')
    print(f"\nSaved figure to: {output_path}")
    print(f"Figure size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    # paths
    project_root = Path(__file__).parent.parent
    tb_log_dir = project_root / 'data' / 'models' / 'runs' / 'ii-cl-19m_v1.0' / 'tensorboard'
    output_path = project_root / 'src' / 'web' / 'static' / 'images' / 'training_curves.svg'

    # generate figure
    plot_training_curves(tb_log_dir, output_path)
