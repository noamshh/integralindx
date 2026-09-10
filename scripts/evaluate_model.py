"""
usage:
    python scripts/evaluate_model.py --checkpoint data/models/checkpoints/egen_v1/best_model.pt
    python scripts/evaluate_model.py --checkpoint data/models/checkpoints/egen_v1/best_model.pt --visualize
    python scripts/evaluate_model.py --checkpoint data/models/checkpoints/baseline/best_model.pt --model ii-cl-400k --training cpu
"""
import argparse
import json
import logging
import random
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.egen.contrastive_model import Encoder
from src.models.egen.tokenizer import Tokenizer
from src.models.egen.datasets.contrastive_dataset import ContrastiveDataset
from src.models.egen.training.losses import build_criterion
from src.models.egen.training.metrics import compute_embedding_quality
from src.models.egen.training.visualization import plot_embedding_space
from src.utils.paths import get_paths

logger = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_intra_sim = 0.0
    total_inter_sim = 0.0
    total_separation = 0.0
    num_batches = len(test_loader)
    all_query_embeddings = []
    all_positive_embeddings = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="evaluating test set")
        for batch_idx, batch in enumerate(pbar):
            query = batch['query'].to(device)
            positive = batch['positive'].to(device)
            negatives = batch['negatives'].to(device)
            query_mask = batch['query_mask'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            negatives_mask = batch['negatives_mask'].to(device)
            # forward
            query_hidden = model(query, mask=query_mask)
            positive_hidden = model(positive, mask=positive_mask)
            query_emb = model.mean_pool(query_hidden, mask=query_mask)
            positive_emb = model.mean_pool(positive_hidden, mask=positive_mask)
            # negatives
            batch_size, n_negatives, seq_len = negatives.shape
            negatives_flat = negatives.view(batch_size * n_negatives, seq_len)
            negatives_mask_flat = negatives_mask.view(batch_size * n_negatives, seq_len)
            negatives_hidden = model(negatives_flat, mask=negatives_mask_flat)
            negatives_emb_flat = model.mean_pool(negatives_hidden, mask=negatives_mask_flat)
            embed_dim = negatives_emb_flat.size(-1)
            negatives_emb = negatives_emb_flat.view(batch_size, n_negatives, embed_dim)
            # loss
            loss = criterion(query=query_emb, pos_key=positive_emb, neg_key=negatives_emb)
            total_loss += loss.item()
            # embedding quality metrics
            quality_metrics = compute_embedding_quality(query_emb, positive_emb, negatives_emb)
            total_intra_sim += quality_metrics['intra_class_similarity']
            total_inter_sim += quality_metrics['inter_class_similarity']
            total_separation += quality_metrics['separation_margin']
            # collect embeddings for visualization
            all_query_embeddings.append(query_emb.cpu())
            all_positive_embeddings.append(positive_emb.cpu())

            # update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'intra': f'{total_intra_sim/(batch_idx+1):.3f}',
                'inter': f'{total_inter_sim/(batch_idx+1):.3f}',
                'sep': f'{total_separation/(batch_idx+1):.3f}'
            })

            # log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"progress: {batch_idx+1}/{num_batches} batches | "
                    f"loss={total_loss/(batch_idx+1):.4f} | "
                    f"separation={total_separation/(batch_idx+1):.3f}"
                )
    # aggregate metrics
    avg_loss = total_loss / num_batches
    avg_intra_sim = total_intra_sim / num_batches
    avg_inter_sim = total_inter_sim / num_batches
    avg_separation = total_separation / num_batches
    # concatenate all embeddings
    all_query_embeddings = torch.cat(all_query_embeddings, dim=0).numpy()
    all_positive_embeddings = torch.cat(all_positive_embeddings, dim=0).numpy()
    results = {
        'test_loss': avg_loss,
        'intra_class_similarity': avg_intra_sim,
        'inter_class_similarity': avg_inter_sim,
        'separation_margin': avg_separation,
        'query_embeddings': all_query_embeddings,
        'positive_embeddings': all_positive_embeddings,
    }
    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"using device: {device}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"loaded checkpoint from {checkpoint_path}")
    tokenizer = Tokenizer()
    vocab_size = len(tokenizer)
    paths = get_paths()

    # load model config
    model_config_path = paths['project_root'] / 'config' / 'model' / f'{args.model}.yaml'
    if not model_config_path.exists():
        raise FileNotFoundError(f"model config not found: {model_config_path}")
    model_cfg = OmegaConf.load(model_config_path)
    encoder_cfg = model_cfg.encoder
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
    logger.info(f"loaded model weights for {model_cfg.name}")

    # load training config and merge with model config to resolve interpolations
    training_config_path = paths['project_root'] / 'config' / 'training' / f'{args.training}.yaml'
    if not training_config_path.exists():
        raise FileNotFoundError(f"training config not found: {training_config_path}")
    training_cfg = OmegaConf.load(training_config_path)

    # merge configs to resolve interpolations like ${model.version}
    merged_cfg = OmegaConf.create({'model': model_cfg, 'training': training_cfg})

    # allow manual override of test dataset path
    if args.test_tsv:
        test_tsv = Path(args.test_tsv)
        logger.info(f"using manual test dataset: {test_tsv}")
    else:
        test_tsv = Path(OmegaConf.select(merged_cfg, 'training.dataset.test_tsv'))

    if not test_tsv.exists():
        raise FileNotFoundError(f"test dataset not found: {test_tsv}")
    max_seq_len = merged_cfg.training.dataset.max_seq_len
    test_dataset = ContrastiveDataset(tsv_path=str(test_tsv), tokenizer=tokenizer, max_seq_len=max_seq_len)
    logger.info(f"test dataset: {len(test_dataset)} examples")

    # sample if requested
    if args.max_test_samples and args.max_test_samples < len(test_dataset):
        logger.info(f"sampling {args.max_test_samples} examples from test set (seed={args.sample_seed})")
        random.seed(args.sample_seed)
        torch.manual_seed(args.sample_seed)
        indices = random.sample(range(len(test_dataset)), args.max_test_samples)
        test_dataset = Subset(test_dataset, indices)
        logger.info(f"using {len(test_dataset)} sampled examples")

    batch_size = args.batch_size or merged_cfg.training.training.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=test_dataset.dataset.collate_fn if hasattr(test_dataset, 'dataset') else test_dataset.collate_fn)
    criterion = build_criterion(**OmegaConf.to_container(merged_cfg.training.criterion))
    logger.info("starting evaluation...")
    results = evaluate(model, criterion, test_loader, device)

    logger.info("=" * 60)
    logger.info("evaluation results:")
    logger.info(f"test loss: {results['test_loss']:.4f}")
    logger.info(f"intra-class similarity: {results['intra_class_similarity']:.4f}")
    logger.info(f"inter-class similarity: {results['inter_class_similarity']:.4f}")
    logger.info(f"separation margin: {results['separation_margin']:.4f}")
    logger.info("=" * 60)

    # save results to JSON
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / 'evaluation'
    output_dir.mkdir(exist_ok=True, parents=True)

    results_json = {
        'checkpoint': str(checkpoint_path),
        'model_config': args.model,
        'training_config': args.training,
        'test_dataset': str(test_tsv),
        'num_test_samples': len(test_dataset) if not hasattr(test_dataset, 'dataset') else len(test_dataset.dataset),
        'num_evaluated': len(results['query_embeddings']),
        'metrics': {
            'test_loss': results['test_loss'],
            'intra_class_similarity': results['intra_class_similarity'],
            'inter_class_similarity': results['inter_class_similarity'],
            'separation_margin': results['separation_margin'],
        }
    }

    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"saved results to {results_path}")

    if args.visualize:
        logger.info("generating embedding visualizations...")
        output_dir = checkpoint_path.parent / 'evaluation'
        output_dir.mkdir(exist_ok=True)
        embeddings = np.vstack([results['query_embeddings'], results['positive_embeddings']])
        labels = ['query'] * len(results['query_embeddings']) + \
                 ['positive'] * len(results['positive_embeddings'])

        # PCA visualization (like E-Gen paper Figure 3)
        logger.info("creating PCA visualization...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(10, 8))
        query_mask = np.array([l == 'query' for l in labels])
        positive_mask = np.array([l == 'positive' for l in labels])
        ax.scatter(embeddings_pca[query_mask, 0], embeddings_pca[query_mask, 1],
                   alpha=0.6, s=20, label='Query', c='blue')
        ax.scatter(embeddings_pca[positive_mask, 0], embeddings_pca[positive_mask, 1],
                   alpha=0.6, s=20, label='Positive', c='red')
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title('PCA Projection of Expression Embeddings', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pca_output = output_dir / 'embedding_space_pca.png'
        plt.savefig(pca_output, dpi=150, bbox_inches='tight')
        logger.info(f"saved PCA visualization to {pca_output}")
        plt.close()

        # t-SNE visualization
        logger.info("creating t-SNE visualization...")
        output_path = output_dir / 'embedding_space_tsne.png'
        plot_embedding_space(
            embeddings=embeddings,
            labels=labels,
            output_path=output_path,
            method='tsne',
            perplexity=min(30, len(embeddings) // 3),
        )
        logger.info(f"saved t-SNE visualization to {output_path}")
    logger.info("evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate trained contrastive model')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--model', type=str, default='ii-cl-19m', help='model config name (default: ii-cl-19m)')
    parser.add_argument('--training', type=str, default='gpu', help='training config name (default: gpu)')
    parser.add_argument('--test-tsv', type=str, default=None, help='path to test dataset TSV (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size for evaluation (default: use training config)')
    parser.add_argument('--max-test-samples', type=int, default=None, help='max number of test samples to evaluate (default: all)')
    parser.add_argument('--sample-seed', type=int, default=42, help='random seed for sampling (default: 42)')
    parser.add_argument('--visualize', action='store_true', help='generate embedding space visualization')
    parser.add_argument('--output-dir', type=str, default=None, help='output directory for results (default: checkpoint_dir/evaluation)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
