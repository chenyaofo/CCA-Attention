import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import types

from .cca_pooling import _weight_pooling, fast_calculate_fusion_weights
from .cca import cca_attention_v2

import transformers
from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaForCausalLM

logger = logging.get_logger(__name__)



def simple_pooling_forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		cache_position: Optional[torch.LongTensor] = None,
		position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
		**kwargs,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		bsz, q_len, _ = hidden_states.size()
		logger.warning_once(
			"------------------this is in customized attention forward--------------"
		)
		if self.config.pretraining_tp > 1:
			key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
			query_slices = self.q_proj.weight.split(
				(self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
			)
			key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
			value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

			query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
			query_states = torch.cat(query_states, dim=-1)

			key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
			key_states = torch.cat(key_states, dim=-1)

			value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
			value_states = torch.cat(value_states, dim=-1)

		else:
			query_states = self.q_proj(hidden_states)
			key_states = self.k_proj(hidden_states)
			value_states = self.v_proj(hidden_states)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

		if position_embeddings is None:
			logger.warning_once(
				"The attention layers in this model are transitioning from computing the RoPE embeddings internally "
				"through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
				"`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
				"removed and `position_embeddings` will be mandatory."
			)
			cos, sin = self.rotary_emb(value_states, position_ids)
		else:
			cos, sin = position_embeddings
		pooling_attention = pooling_attention_w_cache if use_cache and query_states.size(2)==1 else \
			pooling_attention_wo_cache
		attn_output = pooling_attention(query_states, key_states, value_states,
								  pooling_size, window_size, self.attention_dropout,
								  self.num_key_value_groups, self.layer_idx, self.rotary_emb,
								  cos, sin, past_key_value, output_attentions, use_cache, cache_position)

		attn_output = attn_output.transpose(1, 2).contiguous()
		attn_output = attn_output.view(bsz, q_len, -1)

		attn_output = self.o_proj(attn_output)

		return attn_output, None, past_key_value

def generate_mask(seq_len, pooling_size, num_pool, window_size, device='cpu', dtype=torch.bfloat16, cacheing: bool = False):
	'''
	Generate a mask for the attention mechanism.
	
	The token can only attend to the pooling tokens before it and the tokens in a window of window_size.
	'''
	# mask: [seq_len, num_pool + seq_len]
	# for pooling tokens, only the tokens between window size is used
	row_ind = torch.arange(seq_len, device=device, dtype=torch.long)+1
	pool_ind = torch.arange(num_pool, device=device, dtype=torch.long)+1
	pool_mask = (row_ind.unsqueeze(1)>window_size) * (torch.floor_divide(row_ind-window_size, pooling_size).unsqueeze(1) >= pool_ind.unsqueeze(0))
	
	# for non-pooling tokens, only the tokens before it and the remaining tokens after pooling
	col_ind = torch.arange(seq_len, device=device, dtype=torch.long)+1
	# window size is adaptive based on the pooling size and currrent query position
	window_size = torch.tensor([window_size]*seq_len, device=device, dtype=torch.long)
	num_pooling_token = torch.maximum(torch.floor_divide(row_ind-window_size, pooling_size)*pooling_size, torch.tensor([0]*seq_len, device=device, dtype=torch.long))
	remain_num = torch.maximum(row_ind-num_pooling_token-window_size, torch.tensor([0]*seq_len, device=device, dtype=torch.long))
	# remain_num = remain_num.unsqueeze(1)
	window_size = window_size+remain_num
	window_mask = (row_ind.unsqueeze(1)>=col_ind.unsqueeze(0)) * ((row_ind-window_size).unsqueeze(1)<col_ind.unsqueeze(0))
	bool_mask = torch.concat([pool_mask, window_mask], dim=1)
	final_mask = torch.zeros((seq_len, seq_len+num_pool), device=device, dtype=dtype)
	final_mask = torch.masked_fill(final_mask, ~bool_mask, -torch.inf)
	if cacheing:
		final_mask = final_mask[-1:]
	return final_mask


def _attention_impl(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
					attention_dropout: float, seq_len: int, pooling_size: int, num_pool: int, window_size: int):
	'''
	Q: [bsz, num_heads, num_pool + s_len, head_dim]
	K: [bsz, num_heads, num_pool + s_len, head_dim]
	V: [bsz, num_heads, num_pool + s_len, head_dim]
	'''
	attention_mask = None
	if seq_len>window_size and Q.shape[2]>1:
		attention_mask = generate_mask(seq_len, pooling_size, num_pool, window_size, device=Q.device, dtype=Q.dtype, cacheing=Q.shape[2]==1)
	is_causal = True if Q.shape[2]>1 and attention_mask is None else False
	with torch.backends.cuda.sdp_kernel(enable_flash=True):
		output = F.scaled_dot_product_attention(Q, K, V,
										  attn_mask=attention_mask,
										  dropout_p=attention_dropout, is_causal=is_causal)
	return output
	# masking
	

def pooling_attention_wo_cache(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
					  pooling_size: int, window_size:int, attention_dropout: float,
					  num_key_value_groups: int, layer_idx: int, rope_embed,
					  cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None, 
					  past_key_value: Optional[Cache] = None,
					  output_attentions: bool = False,
					  use_cache: bool = False,
					  cache_position: Optional[torch.LongTensor] = None,
					  ):
	'''
	Q: [bsz, num_heads, s_len, head_dim]
	K: [bsz, num_heads, s_len, head_dim]
	V: [bsz, num_heads, s_len, head_dim]
	pooling_size: the size of the pooling window in pre-context
	window_size: the size of the window in slide window attention
	'''
	# pooling and concatenate
	# in inference stage, the seq len may not be multiple of 1024
	# pooling_k, pooling_v = Q, K
	Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
	if use_cache and past_key_value is not None:
		# sin and cos are specific to RoPE models; cache_position needed for the static cache
		cache_kwargs = {"sin": sin, "cos": cos, 'local_size': window_size+pooling_size}
		_, K, V = past_key_value.update_kv(Q,K, V, layer_idx, cache_kwargs)
	K = repeat_kv(K, num_key_value_groups)
	V = repeat_kv(V, num_key_value_groups)
	bsz, nh, k_len, hd = K.shape
	
	q_len = Q.shape[2]
	pooling_len = q_len - window_size # we only pooling the pre-context before fine-grained window attention
	num_pool = pooling_len // pooling_size # perform pooling attention when and only when the length of pre-context is larger than window size at least one group
	if num_pool>0:
		pooling_len = int(num_pool * pooling_size)
		pooling_q, pooling_k, pooling_v = Q[:, :, :pooling_len], \
			K[:, :, :pooling_len], \
			V[:, :, :pooling_len] # [bsz, num_heads, pooling_len, head_dim]
		# import pdb; pdb.set_trace()
		if pooling_func == 'max':
			pooled_k = pooling_k.reshape(bsz, nh, num_pool, pooling_size, -1).max(dim=-2)[0]
			pooled_v = pooling_v.reshape(bsz, nh, num_pool, pooling_size, -1).max(dim=-2)[0]
		elif pooling_func == 'mean':
			pooled_k = pooling_k.reshape(bsz, nh, num_pool, pooling_size, -1).mean(dim=-2)
			pooled_v = pooling_v.reshape(bsz, nh, num_pool, pooling_size, -1).mean(dim=-2)
		elif pooling_func == 'cca':
			weights = fast_calculate_fusion_weights(pooling_q.transpose(1,2),
													pooling_k.transpose(1,2),
													num_pool, nh)
			# [bsz, num_pool, num_heads, pooling_size] -> [bsz, num_heads, num_pool, pooling_size, 1]
			weights = weights.permute(0,2,1,3).unsqueeze(-1).contiguous()

			pooling_k, pooling_v = pooling_k.transpose(1,2).reshape(bsz, pooling_len, -1),\
				pooling_v.transpose(1,2).reshape(bsz, pooling_len, -1)

			# [bsz, num_pool, dim]
			pooled_k, pooled_v = _weight_pooling(weights, pooling_k, pooling_v, pooling_size, nh)

			# [bsz, num_pool, dim] -> [bsz, num_heads, num_pool, head_dim]
			pooled_k, pooled_v = pooled_k.reshape(bsz, num_pool, nh, -1).transpose(1,2).contiguous(),\
				pooled_v.reshape(bsz, num_pool, nh, -1).transpose(1,2).contiguous()
		
		if use_cache and past_key_value is not None:
			# sin and cos are specific to RoPE models; cache_position needed for the static cache
			cache_kwargs = {"sin": sin, "cos": cos, 'local_size': window_size+pooling_size}
			pooled_k, pooled_v = past_key_value.update_group_kv(pooled_k, pooled_v, layer_idx, cache_kwargs)
		
		window_length = K.shape[2]
		# concat the pooled and window k,v
		K = torch.cat([pooled_k, K], dim=2)
		V = torch.cat([pooled_v, V], dim=2)
	if q_len < 300:
		output = _attention_impl(Q, K, V, 
							attention_dropout, 
							k_len, pooling_size,
							num_pool, window_size)
	else:
		output = cca_attention_v2(Q,K,V, causal=True, pool_size=pooling_size, window_size=window_size)
	return output

def pooling_attention_w_cache(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
							pooling_size: int, window_size:int, attention_dropout: float,
							num_key_value_groups: int, layer_idx: int, rope_embed,
							cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None, 
							past_key_value: Optional[Cache] = None,
							output_attentions: bool = False,
							use_cache: bool = False,
							cache_position: Optional[torch.LongTensor] = None,
							):
	assert use_cache and past_key_value is not None, "This is only for kv cache, if not, use pooling_attention_wo_cache instead"
	Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
	cache_kwargs = {"sin": sin, "cos": cos, 'local_size': window_size+pooling_size}
	pooling_q, K, V = past_key_value.update_kv(Q, K, V, layer_idx, cache_kwargs)
	K = repeat_kv(K, num_key_value_groups)
	V = repeat_kv(V, num_key_value_groups)
	bsz, num_heads, _, _ = K.shape
	pooling_len = past_key_value.seen_tokens - window_size
	num_pool = pooling_len // pooling_size
	if pooling_len%pooling_size==0 and num_pool>0:
		num_pool=1
		pooling_q = pooling_q[:, :, :pooling_size]
		pooling_k, pooling_v = K[:, :, :pooling_size], \
			V[:, :, :pooling_size]
		# Q: [bsz, num_heads, q_len, dim]-> [bsz, num_pool, num_heads, 1, dim]
		# pooling_q = Q[:,:,-1:].unsqueeze(1)
		weights = fast_calculate_fusion_weights(pooling_q.transpose(1,2), pooling_k.transpose(1,2), num_pool, num_heads)
		weights = weights.permute(0,2,1,3).unsqueeze(-1).contiguous()
		pooling_k, pooling_v = pooling_k.transpose(1,2).reshape(bsz, pooling_size, -1),\
			pooling_v.transpose(1,2).reshape(bsz, pooling_size, -1)
		# TODO: check cos and sin is only one group
		pooled_k, pooled_v = _weight_pooling(weights, pooling_k, pooling_v, pooling_size, num_heads)

		# [bsz, num_pool, dim] -> [bsz, num_heads, num_pool, head_dim]
		pooled_k, pooled_v = pooled_k.reshape(bsz, num_pool, num_heads, -1).transpose(1,2).contiguous(),\
			pooled_v.reshape(bsz, num_pool, num_heads, -1).transpose(1,2).contiguous()
		
		position_ids = torch.arange(past_key_value.seen_tokens-window_size-pooling_size, past_key_value.seen_tokens-window_size, dtype=torch.long, device=Q.device).unsqueeze(0).repeat(bsz,1)
		cos, sin = rope_embed(pooling_k, position_ids)
		cache_kwargs = {"sin": sin, "cos": cos, 'local_size': window_size+pooling_size}
		pooled_k, pooled_v= past_key_value.update_group_kv(pooled_k, pooled_v, layer_idx, cache_kwargs)
		
		# concat the pooled and window k,v
		K = K[:,:, -window_size:]
		V = V[:,:, -window_size:]
		K = torch.cat([pooled_k, K], dim=2)
		V = torch.cat([pooled_v, V], dim=2)
	elif num_pool>0:
		remain_num = past_key_value.seen_tokens - num_pool*pooling_size
		pooled_k, pooled_v = past_key_value.group_key[layer_idx], \
			past_key_value.group_value[layer_idx]
		K = K[:,:, -remain_num:]
		V = V[:,:, -remain_num:]
		# concat the pooled and window k,v
		K = torch.cat([pooled_k, K], dim=2)
		V = torch.cat([pooled_v, V], dim=2)
	
	output = _attention_impl(Q, K, V, 
						  attention_dropout, 
						  past_key_value.seen_tokens, pooling_size,
						  num_pool, window_size)
	return output

def pooling_attention_w_cache_full_cache(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
							pooling_size: int, window_size:int, attention_dropout: float,
							num_key_value_groups: int, layer_idx: int, rope_embed,
							cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None, 
							past_key_value: Optional[Cache] = None,
							output_attentions: bool = False,
							use_cache: bool = False,
							cache_position: Optional[torch.LongTensor] = None,
							):
	assert use_cache and past_key_value is not None, "This is only for kv cache, if not, use pooling_attention_wo_cache instead"
	Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
	cache_kwargs = {"sin": sin, "cos": cos, 'local_size': -1}
	pooling_q, K, V = past_key_value.update_kv(Q, K, V, layer_idx, cache_kwargs)
	K = repeat_kv(K, num_key_value_groups)
	V = repeat_kv(V, num_key_value_groups)
	bsz, num_heads, _, _ = K.shape
	pooling_len = past_key_value.seen_tokens - window_size
	num_pool = pooling_len // pooling_size
	output = _attention_impl(Q, K, V, 
						  attention_dropout, 
						  past_key_value.seen_tokens, pooling_size,
						  num_pool, window_size)
	return output

def replace_attn_forward(p_size = 16, w_size = 1024, pool_func='cca'):
	global pooling_size
	global window_size
	global pooling_func
	pooling_size = p_size
	window_size = w_size
	pooling_func = pool_func
	assert pooling_func in ['cca', 'mean', 'max'], 'only support cca, mean, max pooling'
	
	transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = simple_pooling_forward
	transformers.models.llama.modeling_llama.LlamaAttention.forward = simple_pooling_forward
	transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = simple_pooling_forward
	print(f'replace attention forward with simple pooling forward {pooling_size=}, {window_size=}, {pooling_func=}')

if __name__ == '__main__':
	seq_len = 16
	pooling_size = 1
	window_size = 4
	num_pool = (seq_len-window_size)//pooling_size
	# test
	print(generate_mask(seq_len, pooling_size, num_pool,window_size))