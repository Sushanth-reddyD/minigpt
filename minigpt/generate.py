import torch


def generate(
    model,
    token_ids,
    max_new_tokens,
    context_length,
    temperature=0.0,
    top_k=None,
    top_p=None,
):
    for _ in range(max_new_tokens):
        context = token_ids[:, -context_length:]
        logits = model(context)
        next_logits = logits[:, -1, :]

        if temperature > 0.0:
            next_logits = next_logits / temperature

            if top_k is not None:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_val,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                cutoff_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[cutoff_mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_ids = torch.cat([token_ids, next_id], dim=1)
    return token_ids
