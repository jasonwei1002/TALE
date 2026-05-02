import os
from typing import Type
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm

def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
def clip_loss(x, y, temperature=0.07, device='cuda'):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels) 
    loss_i = F.cross_entropy(sim.T, labels) 

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i), acc1, acc5


def _jaccard_matrix(labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    labels = (labels > 0).to(dtype=torch.float32)
    inter = labels @ labels.T
    row_sum = labels.sum(dim=1, keepdim=True)
    union = row_sum + row_sum.T - inter
    return inter / (union + eps)


def _multi_positive_nce(sim: torch.Tensor,
                        pos_mask: torch.Tensor,
                        soft_neg_mask: torch.Tensor,
                        soft_neg_scale: float,
                        eps: float = 1e-8) -> torch.Tensor:
    exp_sim = torch.exp(sim)
    pos_mask_f = pos_mask.to(dtype=exp_sim.dtype)
    soft_neg_mask_f = soft_neg_mask.to(dtype=exp_sim.dtype)
    neg_mask_f = (~pos_mask & ~soft_neg_mask).to(dtype=exp_sim.dtype)

    denom = (exp_sim * (pos_mask_f + neg_mask_f + soft_neg_scale * soft_neg_mask_f)).sum(dim=1)
    num = (exp_sim * pos_mask_f).sum(dim=1)
    loss = -torch.log((num + eps) / (denom + eps))
    return loss.mean()


def clip_loss_jaccard(x,
                      y,
                      labels,
                      temperature=0.07,
                      jaccard_t=0.2,
                      soft_neg_scale=0.5,
                      device='cuda'):
    # τ=0 means no Jaccard soft-negative → fall back to standard CLIP loss
    if jaccard_t <= 0.0:
        return clip_loss(x, y, temperature=temperature, device=device)

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

    labels = labels.to(device=device)
    jaccard = _jaccard_matrix(labels)
    pos_mask = jaccard >= jaccard_t
    pos_mask.fill_diagonal_(True)
    soft_neg_mask = (jaccard > 0) & (jaccard < jaccard_t)

    loss_t = _multi_positive_nce(sim, pos_mask, soft_neg_mask, soft_neg_scale)
    loss_i = _multi_positive_nce(sim.T, pos_mask.T, soft_neg_mask.T, soft_neg_scale)

    diag_labels = torch.arange(x.shape[0]).to(device)
    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, diag_labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, diag_labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i), acc1, acc5


def local_contrastive_loss(sent_emb, sent_mask, seg_emb, tau2=5.0, tau3=10.0, device='cuda'):
    sent_emb = F.normalize(sent_emb, dim=-1)
    seg_emb = F.normalize(seg_emb, dim=-1)

    sim = torch.einsum('b m d, j k d -> b j m k', sent_emb, seg_emb)
    att = F.softmax(sim / tau2, dim=-1)
    ctx = torch.einsum('b j m k, j k d -> b j m d', att, seg_emb)
    ctx = F.normalize(ctx, dim=-1)

    sent_ctx_sim = torch.einsum('b m d, b j m d -> b j m', sent_emb, ctx)
    if sent_mask is not None:
        mask = sent_mask[:, None, :].to(dtype=sent_ctx_sim.dtype)
        sent_ctx_sim = sent_ctx_sim.masked_fill(mask == 0, float('-inf'))

    local_sim = tau3 * torch.logsumexp(sent_ctx_sim / tau3, dim=-1)
    local_sim = local_sim / tau2

    labels = torch.arange(local_sim.shape[0]).to(device)
    loss_t = F.cross_entropy(local_sim, labels)
    loss_i = F.cross_entropy(local_sim.T, labels)
    return loss_t + loss_i


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def gloria_attention_fn_1d(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    sourceL = context.size(2)

    contextT = torch.transpose(context, 1, 2).contiguous()

    attn = torch.bmm(contextT, query)
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn


def gloria_local_loss(
    img_tokens,
    text_tokens,
    text_mask=None,
    temp1=4.0,
    temp2=5.0,
    temp3=10.0,
    agg="sum",
):
    img_tokens = img_tokens.float()
    text_tokens = text_tokens.float()

    batch_size = img_tokens.shape[0]

    context = img_tokens.permute(0, 2, 1).contiguous()
    words_emb = text_tokens.permute(0, 2, 1).contiguous()

    att_maps = []
    similarities = []

    for i in range(batch_size):
        if text_mask is not None:
            mask_i = text_mask[i]
            word = words_emb[i, :, mask_i]
            words_num = word.shape[-1]
        else:
            word = words_emb[i]
            words_num = word.shape[-1]

        if words_num <= 0:
            word = words_emb[i, :, :1]
            words_num = 1

        word = word.unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)

        weiContext, attn = gloria_attention_fn_1d(word, context, temp1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())

        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()

        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)

    labels = torch.arange(batch_size, device=similarities.device)
    loss0 = F.cross_entropy(similarities, labels)
    loss1 = F.cross_entropy(similarities1, labels)
    return loss0 + loss1
