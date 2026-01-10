# Pare
## Pruning LLMs 
This project is based on these two Nvidia papers:
- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796)

I'll use their best practices. Directly quoting from *Compact Language Models via Pruning and Knowledge Distillation*:
1. To train a family of LLMs, train the largest one and prune+distill iteratively to smaller LLMs.
2. Use (batch=L2, seq=mean) importance estimation for width axes and PPL/BI for depth.
3. Use single-shot importance estimation; iterative provides no benefit.
4. Prefer width pruning over depth for the model scales we consider (≤ 15B).
5. Retrain exclusively with distillation loss using KLD instead of conventional training.
6. Use (logit+intermediate state+embedding) distillation when depth is reduced significantly.
7. Use logit-only distillation when depth isn’t reduced significantly.
8. Prune a model closest to the target size.
9. Perform lightweight retraining to stabilize the rankings of searched pruned candidates.
10. If the largest model is trained using a multi-phase training strategy, it is best to prune and retrain the model obtained from the final stage of training.

I'll be starting with [Olmo 3 7B Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct), since all the datasets are open sourced. This model is also nice because it's actually not that deep-- it has a similar depth to models that are roughly half its size.

Note that only pruning width seems to slightly defeat the purpose of making smaller models, in my opinion. Nvidia has released a paper on exactly this: [Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](https://arxiv.org/pdf/2511.18890): "While previous work on SLM design has primarily focused on reducing the number of parameters to achieve parameter-optimal SLMs, parameter efficiency does not necessarily translate into proportional real-device speed-ups...we first study latency-optimal depth-width ratios, with the key finding that although deep-thin models generally achieve better accuracy under the same parameter budget, they may not lie on the accuracy-latency trade-off frontier."

I ran the Olmo 3 evals on my machine (I had to edit [olmes](https://github.com/allenai/olmes.git) a fair amount to actually get it running). Here are the baseline stats I'm going with, they match the reported evals well!

## Project Structure
The hope for this project is for it to be really simple, and as flat as possible. Just a few scripts that are hopefully pretty configurable. Ideally, we won't have tons of configs cli args or anything.
First, analyze the width importance and layer importance with importance_analysis.py:
```uv run importance_analysis.py```
This will save a .pt file with the following structure:

| Key | Shape | Description |
|-----|-------|-------------|
| `mlp` | `[n layers, intermediate size]` | Per-neuron importance scores (L2 norm of activations) for each layer's FFN |
| `attention` | `[n layers, number of attention heads]` | Per-head importance scores for each layer |
| `attn_ln` | `[hidden size]` | Aggregated importance per hidden dimension for attention layer norms |
| `ffn_ln` | `[hidden size]` | Aggregated importance per hidden dimension for FFN layer norms |
| `layer` | dict of n layers scalars | Cosine-similarity-based importance per layer (for depth pruning) |

Then, prune!
```uv run prune.py```
Then, evaluate with your eval harness and dataset of choice. I used olmes like I said above, but I actually wouldn't recommend it. (Openbench)[https://github.com/groq/openbench] with vLLM is much less of a hassle, in my opinion. I used olmes to make the comparison to the published Olmo 3 results easier for me. 

Finally, distill and evaluate!
```uv run distill.py```

## Best Practices Notes
2. A pretty important detail: take the mean over sequence length first, and then l2 norm. 
3. One run through the model with 1024 samples is enough to figure out what to trim.
4. See note about width pruning above.

## Open Questions / Future Experiments

### Distillation Strategy
- **On-policy vs off-policy distillation** for pruned model recovery - student generates, teacher scores vs both see same input
- **Larger teacher with top-k logits** (32B OLMo) vs same-size teacher (7B) with full logits - richer signal, cheaper storage? Storing Top-K could be nice

### Pruning Decisions
- **Protect full attention layers?** Layer 27 (full attn) scored lowest on depth importance - drop it or preserve for long-range dependencies?
- **Global vs per-layer MLP pruning** - normalize scores then rank globally, letting some layers keep more neurons than others
- **Prune MHA → GQA?** Mean-pool K/V heads within groups, use importance scores to decide which heads to cluster

### Architecture Search
- **Zero-shot PPL as filter** - prune, eval perplexity, drop catastrophic configs before any training
- **Minimum tokens to see separation** - 20M? 50M? 100M? Funnel approach vs flat search
- **Convergent evidence** - layers weak on both depth score AND global neuron count (24-27) are strong drop candidates

### OLMo3-Specific
- **Sliding vs full attention layer behavior** - do they score differently? Should pruning strategy differ?
- Layer 2 anomaly: decent depth score (15.44) but only 4050 neurons globally - doing something specific with few neurons?
