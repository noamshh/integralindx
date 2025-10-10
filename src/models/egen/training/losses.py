from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean') -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query: Tensor, pos_key: Tensor, neg_key: Tensor) -> Tensor:
        """
        Args:
            query: query embeddings [B, D]
            pos_key: positive key embeddings (equivalent expressions) [B, D]
            neg_key: negative key embeddings (non-equivalent expressions) [B, N_neg, D]
        Returns:
            loss: scalar tensor
        Shape notation:
            B = batch size
            D = embedding dimension
            N_neg = number of negatives per query
        Implementation taken from E-Gen reference: https://github.com/hongbozheng/transformer/criterion.py
        """
        query = F.normalize(input=query, p=2.0, dim=-1, eps=1e-12)
        pos_key = F.normalize(input=pos_key, p=2.0, dim=-1, eps=1e-12)
        neg_key = F.normalize(input=neg_key, p=2.0, dim=-1, eps=1e-12)
        pos_logit = torch.sum(query * pos_key, dim=1, keepdim=True)
        query = query.unsqueeze(dim=1)
        neg_logit = query @ neg_key.transpose(dim0=-2, dim1=-1)
        neg_logit = neg_logit.squeeze(dim=1)
        logits = torch.cat(tensors=[pos_logit, neg_logit], dim=1)
        # labels: positive is always first (index 0)
        labels = torch.zeros(logits.size(dim=0), dtype=torch.int64, device=query.device)
        # InfoNCE = -log(exp(pos/T) / (exp(pos/T) + sum(exp(neg/T))))
        loss = F.cross_entropy(input=logits / self.temperature, target=labels, reduction=self.reduction)
        return loss



def build_criterion(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == 'infonce':
        return InfoNCE(temperature=kwargs.get('temperature', 0.1), reduction=kwargs.get('reduction', 'mean'))
    else:
        raise ValueError(f"Invalid criterion: {name}. Choose from {{'infonce', 'triplet'}}")
