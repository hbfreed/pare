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
The hope for this project is for it to be really simple, and as flat as possible. Just a few scripts that are quite configurable. Ideally, we won't have tons of configs cli args or anything.
First, analyze the width importance and layer importance with importance_analysis.py:
```uv run importance_analysis.py```
This will save a json file (?).
Then, prune!
```uv run prune.py```
Then, evaluate with your eval harness and dataset of choice. I used olmes like I said above, but I actually wouldn't recommend it. (Openbench)[https://github.com/groq/openbench] with vLLM is much less of a hassle, in my opinion. I used olmes to make the comparison to the published Olmo 3 results easier for me. 

Finally, distill, and evaluate!
```uv run distill.py```

## Best Practices Notes
2. A pretty important detail: take the mean over sequence length first, and then l2 norm. 
3. One run through the model with 1024 samples is enough to figure out what to trim.
4. See note about width pruning above.
