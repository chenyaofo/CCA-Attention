import torch
import math

def _weight_pooling(weight, inter_key_states, inter_value_states, pool_size, num_heads):
    # [bsz, q_len, d] -> [bsz, num_pool, pool_size, num_heads, head_dim] -> [bsz, num_heads, num_pool, pool_size, head_dim]
    bsz, q_len, dim = inter_key_states.shape
    num_pool = q_len // pool_size
    inter_key_states = inter_key_states.reshape(bsz, num_pool, pool_size, num_heads, -1)
    inter_value_states = inter_value_states.reshape(bsz, num_pool, pool_size, num_heads, -1)
    inter_key_states = inter_key_states.permute(0, 3, 1, 2, 4)
    inter_value_states = inter_value_states.permute(0, 3, 1, 2, 4)

    key_pooled = (inter_key_states*weight).sum(dim=-2)
    value_pooled = (inter_value_states*weight).sum(dim=-2)
    key_pooled = key_pooled.permute(0, 2, 1, 3).reshape(bsz, num_pool, dim)
    value_pooled = value_pooled.permute(0, 2, 1, 3).reshape(bsz, num_pool, dim)
    return key_pooled, value_pooled


def fast_calculate_fusion_weights(q: torch.Tensor, k:torch.Tensor, num_groups: int, num_heads: int):
    b, s, num_heads, head_size = q.shape
    assert s % num_groups == 0, f"sequence length {s} must be divisible by num_groups{num_groups}, {q.shape}, {k.shape}"

    group_size = s // num_groups

    q = q.view(b, num_groups, group_size, num_heads, head_size)
    k = k.view(b, num_groups, group_size, num_heads, head_size)
    q = q.transpose(2,3) # b, num_groups, num_heads, group_size, head_size
    k = k.transpose(2,3) # b, num_groups, num_heads, group_size, head_size
 
    group_vector = q[:,:,:,-1:,:] # b, num_groups, num_heads, 1, head_size

    weight = k * group_vector # b, num_groups, num_heads, group_size, head_size
    weight = weight.sum(dim=-1, keepdim=False) # b, num_groups, num_heads, group_size

    weight = weight / math.sqrt(head_size)
    weight = weight.softmax(dim=-1) # b, num_groups, num_heads, group_size

    return weight # b, num_groups, num_heads, group_size
