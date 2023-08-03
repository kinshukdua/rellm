
import torch
from transformers import LogitsProcessor


class LogitsMask(LogitsProcessor):
    """
    LogitsMask is a LogitsProcessor that masks logits for tokens that are 
    not in the allowed token ids set.
    """
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = list(set(allowed_token_ids))

    def __call__(self, input_ids, scores):
        device = scores.device
        mask = torch.ones_like(scores) * -1e10
        mask[:, self.allowed_token_ids] = 0
        scores = scores + mask 
        return scores.to(device)
