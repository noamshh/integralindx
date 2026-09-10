
"""
Test lazy loading dataset implementation

Tests:
1. Correctness: lazy loading produces identical results to in-memory mode
2. Memory usage: lazy loading uses significantly less memory
3. Multi-worker compatibility: works with DataLoader num_workers > 0
"""
import pytest
import sys
from pathlib import Path
import psutil
import os

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.egen.datasets.contrastive_dataset import ContrastiveDataset
from src.models.egen.tokenizer import Tokenizer
from torch.utils.data import DataLoader


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def test_tsv_path():
    """use existing test data"""
    path = project_root / "data" / "ml_training" / "test" / "train.tsv"
    if not path.exists():
        pytest.skip(f"test data not found: {path}")
    return str(path)


def get_memory_usage_mb():
    """get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_lazy_vs_memory_correctness(tokenizer, test_tsv_path):
    """verify lazy loading produces identical results to in-memory loading"""
    # load both modes
    dataset_memory = ContrastiveDataset(
        tsv_path=test_tsv_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        in_memory=True
    )

    dataset_lazy = ContrastiveDataset(
        tsv_path=test_tsv_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        in_memory=False
    )

    # verify same length
    assert len(dataset_memory) == len(dataset_lazy), \
        f"length mismatch: in-memory={len(dataset_memory)}, lazy={len(dataset_lazy)}"

    print(f"\nDataset size: {len(dataset_memory)} examples")

    # test random samples (not all 630k lines - that would take too long)
    import random
    sample_indices = random.sample(range(len(dataset_memory)), min(100, len(dataset_memory)))

    for idx in sample_indices:
        item_memory = dataset_memory[idx]
        item_lazy = dataset_lazy[idx]

        # verify identical tokenization
        assert item_memory['query'] == item_lazy['query'], \
            f"query mismatch at index {idx}"
        assert item_memory['positive'] == item_lazy['positive'], \
            f"positive mismatch at index {idx}"
        assert len(item_memory['negatives']) == len(item_lazy['negatives']), \
            f"negatives count mismatch at index {idx}"
        for i, (neg_mem, neg_lazy) in enumerate(zip(item_memory['negatives'], item_lazy['negatives'])):
            assert neg_mem == neg_lazy, \
                f"negative {i} mismatch at index {idx}"

    print(f"✓ Verified {len(sample_indices)} samples - all identical")


def test_lazy_memory_usage(tokenizer, test_tsv_path):
    """verify lazy loading uses significantly less memory"""
    import gc

    # baseline memory
    gc.collect()
    baseline_mb = get_memory_usage_mb()
    print(f"\nBaseline memory: {baseline_mb:.1f} MB")

    # test in-memory mode
    gc.collect()
    mem_before = get_memory_usage_mb()
    dataset_memory = ContrastiveDataset(
        tsv_path=test_tsv_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        in_memory=True
    )
    mem_after = get_memory_usage_mb()
    memory_mode_mb = mem_after - mem_before
    print(f"In-memory mode: {memory_mode_mb:.1f} MB ({len(dataset_memory)} examples)")

    del dataset_memory
    gc.collect()

    # test lazy mode
    gc.collect()
    lazy_before = get_memory_usage_mb()
    dataset_lazy = ContrastiveDataset(
        tsv_path=test_tsv_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        in_memory=False
    )
    lazy_after = get_memory_usage_mb()
    lazy_mode_mb = lazy_after - lazy_before
    print(f"Lazy mode: {lazy_mode_mb:.1f} MB ({len(dataset_lazy)} examples)")

    # calculate savings
    savings_mb = memory_mode_mb - lazy_mode_mb
    savings_pct = (savings_mb / memory_mode_mb * 100) if memory_mode_mb > 0 else 0
    print(f"Memory savings: {savings_mb:.1f} MB ({savings_pct:.1f}%)")

    # lazy mode should use significantly less memory
    assert lazy_mode_mb < memory_mode_mb, \
        f"lazy mode ({lazy_mode_mb:.1f} MB) should use less memory than in-memory mode ({memory_mode_mb:.1f} MB)"

    # should save at least 50% for large datasets
    if len(dataset_lazy) > 10000:
        assert savings_pct > 50, \
            f"expected >50% memory savings for large dataset, got {savings_pct:.1f}%"

    print(f"✓ Lazy loading uses {savings_pct:.1f}% less memory")


def test_dataloader_multiworker(tokenizer, test_tsv_path):
    """verify lazy loading works with DataLoader multi-worker mode"""
    dataset = ContrastiveDataset(
        tsv_path=test_tsv_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        in_memory=False
    )

    # test with multiple workers (simulates real training)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,  # test multi-process loading
        collate_fn=dataset.collate_fn
    )

    print(f"\nTesting DataLoader with num_workers=2...")

    # load a few batches
    batches_loaded = 0
    for batch in dataloader:
        assert 'query' in batch
        assert 'positive' in batch
        assert 'negatives' in batch
        assert batch['query'].shape[0] > 0  # non-empty batch
        batches_loaded += 1
        if batches_loaded >= 5:  # test first 5 batches
            break

    print(f"✓ Successfully loaded {batches_loaded} batches with multiple workers")


def test_line_offset_index(test_tsv_path):
    """test LineOffsetIndex helper class directly"""
    from src.models.egen.datasets.contrastive_dataset import LineOffsetIndex

    index = LineOffsetIndex(test_tsv_path)

    print(f"\nLineOffsetIndex built for {len(index)} lines")

    # test getting lines
    line_0 = index.get_line(0)
    assert line_0, "first line should not be empty"
    assert '\t' in line_0, "TSV line should contain tabs"

    # test boundary
    last_line = index.get_line(len(index) - 1)
    assert last_line, "last line should not be empty"

    # test out of bounds
    with pytest.raises(IndexError):
        index.get_line(len(index))

    with pytest.raises(IndexError):
        index.get_line(-1)

    print(f"✓ LineOffsetIndex working correctly")


if __name__ == "__main__":
    # run with: python -m pytest tests/test_lazy_dataset.py -v -s
    pytest.main([__file__, "-v", "-s"])
