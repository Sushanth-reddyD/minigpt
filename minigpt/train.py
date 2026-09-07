import math
import torch

from minigpt.generate import generate


def calc_loss_batch(input_ids, target_ids, model, device):
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    logits = model(input_ids)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
    )


def calc_loss_loader(loader, model, device, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    if num_batches == 0:
        return float("nan")

    for i, (input_ids, target_ids) in enumerate(loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_ids, target_ids, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    warmup_steps=20,
    max_lr=5e-4,
    min_lr=5e-5,
    max_norm=1.0,
):
    train_losses = []
    val_losses = []
    track_lrs = []
    global_step = 0
    max_steps = num_epochs * len(train_loader)

    for epoch in range(num_epochs):
        model.train()

        for input_ids, target_ids in train_loader:
            lr = get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

            loss = calc_loss_batch(input_ids, target_ids, model, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(
                        train_loader, model, device, num_batches=eval_iter
                    )
                    val_loss = calc_loss_loader(
                        val_loader, model, device, num_batches=eval_iter
                    )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Step {global_step:>6} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

                context_ids = torch.tensor(
                    [tokenizer.encode(start_context)]
                ).to(device)
                generated_ids = generate(
                    model,
                    context_ids,
                    max_new_tokens=50,
                    context_length=model.pos_emb.weight.shape[0],
                )
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                print(f"  → {generated_text}\n")

                model.train()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lrs": track_lrs,
    }
