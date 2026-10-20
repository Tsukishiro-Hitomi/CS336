import torch
from attention import softmax
from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import tokenizer

def predict(logits: torch.Tensor, top_p = .0) -> torch.Tensor:
    # logits: (1, L, D)

    if top_p == .0:
        # sample the max probability
        return logits[:, -1].argmax(dim=-1, keepdim=True)

    # top-p sampling
    temperature = 0.001
    scaled_logits = logits[:, -1] / temperature
    probs = softmax(scaled_logits, -1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
    keep[..., 0] = True

    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs * keep)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)

    token = torch.multinomial(filtered, num_samples=1)

    return token

def infer(prompt: torch.Tensor,
          model: TransformerLM,
          tokenizer: tokenizer,
          top_p = .0,
          special_token = "<|endoftext|>",
          max_length = 1000
          ):
    special_token_id = tokenizer.encode(special_token)[0]
    for _ in range(max_length):
        logits = model(prompt)
        next_token_idx = predict(logits, top_p)
        prompt = torch.cat([prompt, next_token_idx.reshape((1, 1))], dim=1)

        if next_token_idx.item() == special_token_id:
            break

    return prompt
