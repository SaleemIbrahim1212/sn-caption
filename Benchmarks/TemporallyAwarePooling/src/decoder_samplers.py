

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
    
    return sorted_indices.gather(-1, token)

def top_k_sample(logits, k =  10, temperature = 1.0): 
    k = max(1, min(k, logits.size(-1)))
    scaled = logits / max(temperature, 1e-8)
    topk_vals, topk_idx = torch.topk(scaled, k=k, dim=-1)  
    probs = F.softmax(topk_vals, dim=-1)                    
    sampled_local = torch.multinomial(probs, num_samples=1) 
    sampled = topk_idx.gather(1, sampled_local).squeeze(1)  
    return sampled

'''
TODO : Beam Search , sampling with repetetion penalty
'''