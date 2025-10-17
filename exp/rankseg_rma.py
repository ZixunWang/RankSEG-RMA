import logging
from typing import Union

import torch
import numpy as np

TRUNCATE_PROB = 0.1
logger = logging.getLogger(__name__)


def convert_to_nonoverlap_prob(overlap_predict, prob, **kwargs):
    num_class = overlap_predict.size(0)
    overlap_mask = overlap_predict.sum(0) > 1
    nonoverlap_predict = torch.zeros_like(overlap_predict[0], dtype=torch.uint8)
    assert num_class <= 256, 'num_class should be less than 256, when using uint8'
    for c in range(num_class):
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c
    argmax_mask = prob.argmax(0)
    nonoverlap_predict[overlap_mask] = argmax_mask[overlap_mask].type(torch.uint8)
    return nonoverlap_predict


def convert_to_nonoverlap_weight_prob(overlap_predict, prob, **kwargs):
    num_class = overlap_predict.size(0)
    overlap_mask = overlap_predict.sum(0) > 1
    nonoverlap_predict = torch.zeros_like(overlap_predict[0], dtype=torch.uint8)
    assert num_class <= 256, 'num_class should be less than 256, when using uint8'
    weight_prob = prob
    for c in range(num_class):
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c
        volume = safe_to_predict.sum().item()
        weight_prob[c] /= 1 + volume
    argmax_mask = weight_prob.argmax(0)
    nonoverlap_predict[overlap_mask] = argmax_mask[overlap_mask].type(torch.uint8)
    return nonoverlap_predict


def convert_to_nonoverlap_rma_score(overlap_predict, prob, opt_metric, sorted_prob, opt_tau, pb_mean, **kwargs):
    num_class = overlap_predict.size(0)
    nonoverlap_predict = torch.zeros_like(overlap_predict[0], dtype=torch.uint8)
    assert num_class <= 256, 'num_class should be less than 256, when using uint8'
    overlap_mask = overlap_predict.sum(0) > 1
    dim = overlap_predict.size(1)
    upper_bound_scale = (dim + 1) / dim
    increment_score = torch.zeros_like(prob, dtype=torch.float32)
    for c in range(num_class):
        if sorted_prob[c][0] <= TRUNCATE_PROB:  # TODO: review this prune
            continue
        safe_to_predict = overlap_predict[c] & ~overlap_mask
        nonoverlap_predict[safe_to_predict] = c
        mu = prob[c][safe_to_predict].sum().item()
        opt_tau_this_c = safe_to_predict.sum().item()
        if opt_metric == 'dice':
            increment_score[c] = 2 * ((mu + prob[c]) / (opt_tau_this_c + pb_mean[c] + 2) - mu / (opt_tau_this_c + pb_mean[c] + 1))  # lower bound
            # increment_score[c] = 2 * ((mu + prob[c]) / (opt_tau[c] + upper_bound_scale*pb_mean[c] + 1) - mu / (opt_tau[c] + upper_bound_scale*pb_mean[c]))  # upper bound
        else:
            increment_score[c] = (mu + prob[c]) / (opt_tau_this_c + pb_mean[c] - mu - prob[c] + 1) - mu / (opt_tau_this_c + pb_mean[c] - mu)  # lower bound
            # increment_score[c] = (mu + prob[c]) / (opt_tau[c] + upper_bound_scale*pb_mean[c] - mu - prob[c] + 1) - mu / (opt_tau[c] + upper_bound_scale*pb_mean[c] - mu)  # upper bound
    increment_argmax_mask = increment_score.argmax(0)
    nonoverlap_predict[overlap_mask] = increment_argmax_mask[overlap_mask].type(torch.uint8)
    return nonoverlap_predict


def rankseg_rma(
        prob: Union[torch.Tensor, np.ndarray],
        opt_metric: str='dice',
        allow_overlap: bool=False,
        in_batch: bool=True,
        return_tau: bool=False,
        to_nonverlap_method: str='rma_score',
    ) -> Union[torch.Tensor, tuple]:

    is_binary = (prob.shape[1] == 2) and not allow_overlap

    # Check input and convert to tensor if needed
    if isinstance(prob, np.ndarray):
        prob = torch.from_numpy(prob)
    prob = prob.float()

    # check if required memory too large (~3GB), then use cpu
    if prob.nelement() * prob.element_size() > 3e9:  # TODO: review this threshold
        prob = prob.cpu()
        logger.warning(f'Input tensor is too large: {prob.nelement() * prob.element_size() / 1e9:.2f} GB; use CPU instead')

    batch_size = prob.shape[0] if in_batch else 1
    prob = prob.unsqueeze(0) if not in_batch else prob

    if is_binary:
        prob = prob[:, 1:2, ...]
        num_class = 1

    assert opt_metric in ['iou', 'dice'], 'opt_metric should be iou or dice'

    if to_nonverlap_method == 'prob':
        to_nonverlap_fn = convert_to_nonoverlap_prob
    elif to_nonverlap_method == 'weight_prob':
        to_nonverlap_fn = convert_to_nonoverlap_weight_prob
    elif to_nonverlap_method == 'rma_score':
        to_nonverlap_fn = convert_to_nonoverlap_rma_score
    else:
        raise ValueError(f'Invalid to_nonverlap_method: {to_nonverlap_method}')
    
    device = prob.device
    num_class = prob.shape[1]
    img_shape = prob.shape[2:]

    prob = torch.flatten(prob, start_dim=2, end_dim=-1)
    dim = prob.shape[-1]

    predict = torch.zeros(batch_size, num_class, dim, dtype=torch.bool, device=device)

    sorted_prob, top_index = torch.sort(prob, dim=-1, descending=True)
    pb_mean = prob.sum(dim=-1)
    cumsum_prob = torch.cumsum(sorted_prob, dim=-1)

    # Compute optimal tau and cutpoint
    opt_tau = compute_opt_tau(opt_metric, pb_mean, cumsum_prob, dim, device)
    # logger.info(f"opt_tau: {opt_tau}, cutpoint: {cutpoint}")

    for b in range(batch_size):
        for c in range(num_class):
            if sorted_prob[b, c, 0] <= TRUNCATE_PROB:  # TODO: review this prune
                continue
            predict[b, c, top_index[b, c, :opt_tau[b, c]]] = True

    if allow_overlap:
        predict = predict.reshape(batch_size, num_class, *img_shape) if in_batch else predict.squeeze(0).reshape(num_class, *img_shape)
    else:
        if is_binary:
            nonoverlap_predict = predict[:, 0, ...].long()
        else:
            nonoverlap_predict = torch.zeros(batch_size, dim, dtype=torch.uint8, device=device)
            for b in range(batch_size):
                nonoverlap_predict[b] = to_nonverlap_fn(predict[b], prob=prob[b], opt_metric=opt_metric, sorted_prob=sorted_prob[b], top_index=top_index[b], pb_mean=pb_mean[b], opt_tau=opt_tau[b])
        predict = nonoverlap_predict.reshape(batch_size, *img_shape) if in_batch else nonoverlap_predict.reshape(*img_shape)

    if return_tau:
        return predict, opt_tau
    return predict


def compute_opt_tau(
        opt_metric: str, 
        pb_mean: torch.Tensor,
        cumsum_prob: torch.Tensor,
        dim: int,
        device: torch.device
    ):
    """Compute optimal tau and cutpoint based on the selected metric."""
    if opt_metric == 'dice':
        discount = pb_mean.unsqueeze(-1) + torch.arange(1, dim + 1, device=device).view(1, 1, -1) + 1.0  # lower bound
        # discount = (dim+1)/dim * pb_mean.unsqueeze(-1) + torch.arange(1, dim + 1, device=device).view(1, 1, -1)  # upper bound
        metric_values = 2.0 * cumsum_prob / discount
    else:  # IoU metric
        discount = pb_mean.unsqueeze(-1) - cumsum_prob + torch.arange(1, dim + 1, device=device).view(1, 1, -1)  # lower bound
        # discount = (dim+1)/dim * (pb_mean.unsqueeze(-1) - cumsum_prob) + torch.arange(1, dim + 1, device=device).view(1, 1, -1) - 1.0  # upper bound
        metric_values = cumsum_prob / discount

    # Get optimal tau indices
    opt_tau = torch.argmax(metric_values, dim=-1) + 1
    # cutpoint = sorted_prob[torch.arange(batch_size)[:, None], torch.arange(num_class), opt_tau - 1]

    return opt_tau
