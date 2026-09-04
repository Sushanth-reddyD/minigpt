import torch


def generate(model, token_ids, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        context = token_ids[:, -context_length:]
        logits = model(context)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, next_id], dim=1)
    return token_ids
