import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
import sympy as sp
import os


class LineOffsetIndex:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.offsets = []
        self._build_index()
        self._file_cache = {}

    def _build_index(self):
        with open(self.filepath, 'rb') as f:
            offset = 0
            while True:
                self.offsets.append(offset)
                line = f.readline()
                if not line:
                    # remove last offset (EOF)
                    self.offsets.pop()
                    break
                offset = f.tell()

    def _get_file_handle(self):
        pid = os.getpid()
        if pid not in self._file_cache:
            self._file_cache[pid] = open(self.filepath, 'rb')
        return self._file_cache[pid]

    def __len__(self):
        return len(self.offsets)

    def get_line(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(f"line index {idx} out of range [0, {len(self.offsets)})")
        f = self._get_file_handle()
        f.seek(self.offsets[idx])
        line = f.readline().decode('utf-8').strip()
        return line

    def __del__(self):
        for f in self._file_cache.values():
            try:
                f.close()
            except:
                pass


class ContrastiveDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer, max_seq_len: int = 128, in_memory: bool = False):
        """Args:
            tsv_path: path to TSV file
            tokenizer: Tokenizer instance
            max_seq_len: maximum sequence length
            in_memory: load entire dataset into memory (if false, use file seeking)"""
        self.tsv_path = tsv_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.in_memory = in_memory
        if in_memory:
            self._load_in_memory()
        else:
            self._build_index()

    def _load_in_memory(self):
        self.examples = []
        with open(self.tsv_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                parts = [_normalize_for_vocab(part) for part in parts]
                query = parts[0]
                positive = parts[1]
                negatives = parts[2:]
                self.examples.append({'query': query, 'positive': positive, 'negatives': negatives})

    def _build_index(self):
        self.line_index = LineOffsetIndex(self.tsv_path)
        self.examples = None

    def __len__(self):
        if self.in_memory:
            return len(self.examples)
        else:
            return len(self.line_index)

    @staticmethod
    def _parse_line(line: str) -> Dict[str, str]:
        parts = line.split('\t')
        if len(parts) < 3:
            raise ValueError(f"invalid line format: expected at least 3 columns, got {len(parts)}")
        parts = [_normalize_for_vocab(part) for part in parts]
        return {
            'query': parts[0],
            'positive': parts[1],
            'negatives': parts[2:]
        }

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        if self.in_memory:
            example = self.examples[idx]
        else:
            line = self.line_index.get_line(idx)
            if not line:
                raise ValueError(f"empty line at index {idx}")
            example = self._parse_line(line)
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
        """Collate batch of examples with padding.
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


def _normalize_for_vocab(prefix_expr: str) -> str:
    tokens = prefix_expr.split()
    normalized = []
    for token in tokens:
        converted_tokens = _convert_numeric_token(token)
        normalized.extend(converted_tokens)
    return ' '.join(normalized)


def _convert_numeric_token(token: str) -> List[str]:
    try:
        val = int(token)
        if 0 <= val <= 9:
            return [token]
        elif val > 9:
            digits = list(str(val))
            return ["INT+"] + digits
        elif val < 0:
            digits = list(str(abs(val)))
            return ["INT-"] + digits
    except ValueError:
        pass
    try:
        val = float(token)
        rational = sp.nsimplify(val)
        if isinstance(rational, sp.Rational):
            numer_tokens = _convert_numeric_token(str(rational.p))
            denom_tokens = _convert_numeric_token(str(rational.q))
            return ["div"] + numer_tokens + denom_tokens
        elif isinstance(rational, sp.Integer):
            return _convert_numeric_token(str(int(rational)))
    except (ValueError, AttributeError):
        pass
    return [token]
