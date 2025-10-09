import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

class ContrastiveDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer, max_seq_len: int = 128):
        """Args:
            tsv_path: Path to TSV file with contrastive pairs
            tokenizer: MathTokenizer instance
            max_seq_len: Maximum sequence length"""
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []
        with open(tsv_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                query = parts[0]
                positive = parts[1]
                negatives = parts[2:]
                self.examples.append({'query': query, 'positive': positive, 'negatives': negatives})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        example = self.examples[idx]
        query_tokens = self.tokenizer.encode(example['query'], max_len=self.max_seq_len)
        positive_tokens = self.tokenizer.encode(example['positive'], max_len=self.max_seq_len)
        negative_tokens = [
            self.tokenizer.encode(neg, max_len=self.max_seq_len)
            for neg in example['negatives']
        ]
        return {
            'query': query_tokens,
            'positive': positive_tokens,
            'negatives': negative_tokens  # list of token lists
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples with padding.
        Args:
            batch: list of dicts from __getitem__
        Returns:
            dict with tensors:
                - query: [B, L] query token IDs
                - positive: [B, L] positive token IDs
                - negatives: [B, N_neg, L] negative token IDs
                - query_mask: [B, L] True for padding
                - positive_mask: [B, L] True for padding
                - negatives_mask: [B, N_neg, L] True for padding
        """
        pad_id = self.tokenizer.PAD_ID
        queries = [torch.tensor(item['query'], dtype=torch.long) for item in batch]
        positives = [torch.tensor(item['positive'], dtype=torch.long) for item in batch]
        # negatives: each example has N_neg negative sequences
        # batch[i]['negatives'] is a list of N_neg token lists
        negatives_list = []
        for item in batch:
            neg_tensors = [torch.tensor(neg, dtype=torch.long) for neg in item['negatives']]
            negatives_list.append(neg_tensors)
        # pad sequences (pad_sequence pads to max length in batch)
        query_padded = pad_sequence(queries, batch_first=True, padding_value=pad_id)
        positive_padded = pad_sequence(positives, batch_first=True, padding_value=pad_id)
        # pad negatives (need to handle nested structure)
        negatives_flat = [neg for item in negatives_list for neg in item]  # flatten
        negatives_padded = pad_sequence(negatives_flat, batch_first=True, padding_value=pad_id)
        batch_size = len(batch)
        n_negatives = len(negatives_list[0])  # assumes all examples have same number of negatives
        seq_len = negatives_padded.size(1)
        negatives_padded = negatives_padded.view(batch_size, n_negatives, seq_len)
        # create attention masks (True for padding, False for real tokens)
        query_mask = (query_padded == pad_id)
        positive_mask = (positive_padded == pad_id)
        negatives_mask = (negatives_padded == pad_id)
        return {
            'query': query_padded,
            'positive': positive_padded,
            'negatives': negatives_padded,
            'query_mask': query_mask,
            'positive_mask': positive_mask,
            'negatives_mask': negatives_mask,
        }
