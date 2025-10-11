from typing import List
from collections import Counter
from src.models.egen.vocab import CONSTANTS, VARIABLES, OPERATORS


class MathTokenizer:
    """Core logic from E-Gen reference: https://github.com/hongbozheng/transformer/tokenizer.py"""
    def __init__(self):
        self.vocab = {}  # token -> id
        self.id_to_token = {}  # id -> token
        self.PAD_TOKEN = 'PAD'
        self.SOE_TOKEN = 'SOE'
        self.EOE_TOKEN = 'EOE'
        self.PAD_ID = 0
        self.SOE_ID = 1
        self.EOE_ID = 2
        self._init_vocab()

    def _init_vocab(self):
        special_tokens = [self.PAD_TOKEN, self.SOE_TOKEN, self.EOE_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.id_to_token[idx] = token
        next_id = len(special_tokens)
        for token in CONSTANTS:
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
        for token in VARIABLES.keys():
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
        for token in OPERATORS.keys():
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

    def build_vocab_from_file(self, tsv_path: str, min_freq: int = 2):
        token_counts = Counter()
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                for expr in parts:
                    tokens = expr.split()  # space-separated prefix notation
                    token_counts.update(tokens)
        # build vocabulary from frequent tokens
        next_id = len(self.vocab)
        for token, count in token_counts.items():
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

    def encode(self, expression: str, max_len: int = 128) -> List[int]:
        tokens = expression.split()
        token_ids = [self.SOE_ID]
        for token in tokens:
            if token not in self.vocab:
                raise ValueError(
                    f"Unknown token '{token}' in expression. "
                    f"All tokens must be in vocabulary (see vocab.py)"
                )
            token_ids.append(self.vocab[token])
        token_ids.append(self.EOE_ID)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len-1] + [self.EOE_ID]
        while len(token_ids) < max_len:
            token_ids.append(self.PAD_ID)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id == self.EOE_ID:
                break
            if token_id not in [self.PAD_ID, self.SOE_ID]:
                token = self.id_to_token.get(token_id, f'<UNKNOWN_{token_id}>')
                tokens.append(token)
        return ' '.join(tokens)

    def __len__(self):
        return len(self.vocab)
