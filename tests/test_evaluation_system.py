"""test evaluation system integration"""
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

from src.models.egen.contrastive_model import Encoder
from src.models.egen.tokenizer import Tokenizer
from src.models.egen.training.evaluation import EvaluationManager
from src.utils.paths import get_paths


@pytest.fixture
def simple_model():
    """create simple model for testing"""
    tokenizer = Tokenizer()
    vocab_size = len(tokenizer.vocab)
    model = Encoder(
        vocab_size=vocab_size,
        dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        max_seq_len=64,
        dropout=0.1
    )
    return model


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def eval_manager(tokenizer, tmp_path):
    """create evaluation manager with test config"""
    paths = get_paths()
    eval_dir = paths['project_root'] / 'data' / 'eval'
    if not (eval_dir / 'topk_queries.json').exists():
        pytest.skip("Evaluation queries not available")
    eval_config = {
        'topk_queries_path': str(eval_dir / 'topk_queries.json'),
        'analogy_queries_path': str(eval_dir / 'analogy_queries.json'),
        'db_path': str(paths['project_root'] / 'data' / 'integral.db'),
        'warmup_iterations': 1,
        'tier1_every_n_iters': 5,
        'tier2_every_n_iters': 10,
        'tier3_every_n_iters': 20,
        'n_samples_tier2': 100,
        'n_viz_samples': 50,
        'seed': 42,
    }

    manager = EvaluationManager.from_config(
        eval_cfg=eval_config,
        tokenizer=tokenizer,
        embedding_dim=128,
        checkpoint_dir=tmp_path,
    )
    return manager


def test_evaluation_manager_init(eval_manager):
    """test evaluation manager initialization"""
    assert eval_manager is not None
    assert len(eval_manager.eval_queries) == 100
    assert eval_manager.embedding_dim == 128


def test_should_evaluate(eval_manager):
    """test evaluation scheduling logic"""
    # below warmup
    assert not eval_manager.should_evaluate(0, tier=1)

    # tier 1 frequency
    assert eval_manager.should_evaluate(5, tier=1)
    assert not eval_manager.should_evaluate(6, tier=1)

    # tier 2 frequency
    assert eval_manager.should_evaluate(10, tier=2)
    assert not eval_manager.should_evaluate(11, tier=2)

    # tier 3 frequency
    assert eval_manager.should_evaluate(20, tier=3)
    assert not eval_manager.should_evaluate(21, tier=3)


def test_tier1_evaluation(simple_model, eval_manager):
    """test tier 1 evaluation runs without error"""
    device = torch.device('cpu')
    simple_model.eval()

    # run tier 1 evaluation
    metrics = eval_manager.evaluate_tier1(simple_model, device)

    # verify metrics structure
    assert 'recall@1' in metrics
    assert 'recall@5' in metrics
    assert 'recall@10' in metrics
    assert 'mrr' in metrics
    assert 'map' in metrics
    assert 'n_queries' in metrics
    assert 'n_groups' in metrics
    assert 'eval_time_sec' in metrics

    # verify metric ranges
    assert 0.0 <= metrics['recall@1'] <= 1.0
    assert 0.0 <= metrics['recall@5'] <= 1.0
    assert 0.0 <= metrics['recall@10'] <= 1.0
    assert 0.0 <= metrics['mrr'] <= 1.0
    assert 0.0 <= metrics['map'] <= 1.0
    assert metrics['n_queries'] > 0
    assert metrics['n_groups'] > 0
    assert metrics['eval_time_sec'] > 0


def test_tier2_evaluation(simple_model, eval_manager):
    """test tier 2 evaluation runs without error"""
    device = torch.device('cpu')
    simple_model.eval()

    # run tier 2 evaluation
    metrics = eval_manager.evaluate_tier2(simple_model, device)

    # verify metrics structure
    assert 'recall@1' in metrics
    assert 'recall@5' in metrics
    assert 'recall@10' in metrics
    assert 'family_purity@1' in metrics
    assert 'family_purity@5' in metrics
    assert 'family_purity@10' in metrics
    assert 'intra_family_sim' in metrics
    assert 'inter_family_sim' in metrics
    assert 'separation_margin' in metrics

    # verify metric ranges
    assert 0.0 <= metrics['recall@5'] <= 1.0
    assert 0.0 <= metrics['family_purity@5'] <= 1.0
    assert -1.0 <= metrics['intra_family_sim'] <= 1.0
    assert -1.0 <= metrics['inter_family_sim'] <= 1.0


def test_run_evaluation_hooks(simple_model, eval_manager):
    """test consolidated evaluation hooks"""
    device = torch.device('cpu')
    simple_model.eval()

    # run hooks (should execute tier 1 at iteration 5)
    eval_manager.run_evaluation(
        model=simple_model,
        device=device,
        iteration=5,
        tb_logger=None,
        console_logger=None,
    )
    # should not raise any errors


@pytest.mark.slow
def test_tier3_evaluation(simple_model, eval_manager):
    """test tier 3 comprehensive evaluation (slow, loads full DB)"""
    device = torch.device('cpu')
    simple_model.eval()

    # run tier 3 evaluation
    results = eval_manager.evaluate_tier3(simple_model, device, iteration=100)

    # verify results structure
    assert 'metrics' in results
    assert 'search_examples' in results
    assert 'visualization_path' in results

    # verify search examples is a string
    assert isinstance(results['search_examples'], str)
    assert 'Query: sin(x)' in results['search_examples']
    assert 'Query: cos(x)' in results['search_examples']

    # verify metrics
    metrics = results['metrics']
    assert 'recall@1' in metrics
    assert 'recall@5' in metrics
    assert 'recall@10' in metrics
    assert 'mrr' in metrics
    assert 'separation_margin' in metrics
