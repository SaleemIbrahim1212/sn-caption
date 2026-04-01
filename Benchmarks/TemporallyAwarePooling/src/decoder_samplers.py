

import torch
import torch.nn.functional as F


'''
Set of sampling techniques to help LSTM in providing rich(er) captions
Ensemble de techniques d'échantillonnage pour aider le LSTM à fournir des légendes plus riches
'''


def greedy_sample(logits, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-8)
    return torch.argmax(logits, dim=-1)

def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[mask] = float('-inf')
    
    probs = torch.softmax(sorted_logits, dim=-1)
    token = torch.multinomial(probs, 1)
    
    return sorted_indices.gather(-1, token).squeeze(-1)

def top_k_sample(logits, k =  10, temperature = 1.0): 
    k = max(1, min(k, logits.size(-1)))
    scaled = logits / max(temperature, 1e-8)
    topk_vals, topk_idx = torch.topk(scaled, k=k, dim=-1)  
    probs = F.softmax(topk_vals, dim=-1)                    
    sampled_local = torch.multinomial(probs, num_samples=1) 
    sampled = topk_idx.gather(1, sampled_local).squeeze(1)  
    return sampled


@torch.no_grad()
def beam_search_decode(step_fn, start_token, eos_token, device, beam_size=3, max_seq_length=50, length_penalty=0.6):

    beams = [{"tokens": [], "score": 0.0, "state": None, "done": False}]

    for i in range(max_seq_length):
        candidates = []
        for beam in beams:
            if beam["done"]:
                candidates.append(beam)
                continue

            prev = torch.tensor([start_token if len(beam["tokens"]) == 0 else beam["tokens"][-1]], dtype=torch.long, device=device)
            logits, next_state = step_fn(prev, beam["state"])
            log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
            top_logp, top_idx = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

            for j in range(top_idx.numel()):
                tok = int(top_idx[j].item())
                candidates.append({
                    "tokens": beam["tokens"] + [tok],
                    "score": beam["score"] + float(top_logp[j].item()),
                    "state": next_state,
                    "done": (tok == eos_token),
                })

        def rank_fn(b):
            L = max(len(b["tokens"]), 1)
            return b["score"] / (L ** length_penalty)

        candidates.sort(key=rank_fn, reverse=True)
        beams = candidates[:beam_size]
        if all(b["done"] for b in beams):
            break

    best = max(beams, key=lambda b: b["score"] / (max(len(b["tokens"]), 1) ** length_penalty))
    return torch.tensor(best["tokens"], dtype=torch.long, device=device)

'''
TODO : sampling with repetetion penalty
'''
