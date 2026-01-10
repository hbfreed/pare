import torch

importance_scores = torch.load("importance_scores_tensors/importance_scores.pt")
width_pruning_factor = 0.5  # TODO: use a float or list
