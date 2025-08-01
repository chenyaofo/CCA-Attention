from typing import Optional, Tuple, List, Dict, Any
import torch
from transformers.cache_utils import Cache

class DynamicCache(Cache):
	"""
	A cache that grows dynamically as more tokens are generated. This is the default for generative models.

	It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
	`[batch_size, num_heads, seq_len, head_dim]`.
	"""

	def __init__(self) -> None:
		self.query_cache: List[torch.Tensor] = []
		self.key_cache: List[torch.Tensor] = []
		self.value_cache: List[torch.Tensor] = []
		self.group_key: List[torch.Tensor] = []
		self.group_value: List[torch.Tensor] = []
		self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

	def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
		"""
		Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
		sequence length.
		"""
		if layer_idx < len(self):
			return (self.query_cache[layer_idx],self.key_cache[layer_idx], self.value_cache[layer_idx], 
					self.group_key[layer_idx], self.group_value[layer_idx])
		else:
			raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

	def __iter__(self):
		"""
		Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
		keys and values
		"""
		for layer_idx in range(len(self)):
			yield (self.query_cache[layer_idx],self.key_cache[layer_idx], self.value_cache[layer_idx],
				   self.group_key[layer_idx], self.group_value[layer_idx])

	def __len__(self):
		"""
		Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
		to the number of layers in the model.
		"""
		return len(self.key_cache)
	
	def update_kv(self,
			   query_states: torch.Tensor,
			   key_states: torch.Tensor,
			   value_states: torch.Tensor,
			   layer_idx: int,
			   cache_kwargs: Optional[Dict[str, Any]] = None,
			   ) -> Tuple[torch.Tensor, torch.Tensor]:
		if layer_idx == 0:
			self._seen_tokens += key_states.shape[-2]
		if cache_kwargs is not None:
			local_size = cache_kwargs['local_size']
			# pool_size = cache_kwargs['pool_size']
		# Update the cache
		if len(self.key_cache) <= layer_idx:
			self.query_cache.append(query_states)
			self.key_cache.append(key_states)
			self.value_cache.append(value_states)
		else:
			self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
			self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
			self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
			if cache_kwargs is not None and layer_idx not in [0,1]:
				if local_size>0 and self.key_cache[layer_idx].size(-2)>local_size:
					self.query_cache[layer_idx] = self.query_cache[layer_idx][...,-local_size:, :]
					self.key_cache[layer_idx] = self.key_cache[layer_idx][...,-local_size:, :]
					self.value_cache[layer_idx] = self.value_cache[layer_idx][...,-local_size:, :]
			
		return self.query_cache[layer_idx],self.key_cache[layer_idx], self.value_cache[layer_idx]
	
	def update_group_kv(self,
						group_key_states: torch.Tensor,
						group_value_states: torch.Tensor,
						layer_idx: int,
						cache_kwargs: Optional[Dict[str, Any]] = None,
						) -> Tuple[torch.Tensor, torch.Tensor]:
		# Update the cache
		if len(self.group_key) <= layer_idx:
			self.group_key.append(group_key_states)
			self.group_value.append(group_value_states)
			return self.group_key[layer_idx], self.group_value[layer_idx]
		else:
			self.group_key[layer_idx] = torch.cat([self.group_key[layer_idx], group_key_states], dim=-2)
			self.group_value[layer_idx] = torch.cat([self.group_value[layer_idx], group_value_states], dim=-2)
			return self.group_key[layer_idx], self.group_value[layer_idx]
	
	def update(
		self,
		key_states: torch.Tensor=None,
		value_states: torch.Tensor=None,
		layer_idx: int =0,
		cache_kwargs: Optional[Dict[str, Any]] = None,
		query_states: torch.Tensor=None,
		group_key_states: torch.Tensor=None,
		group_value_states: torch.Tensor=None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

		Parameters:
			key_states (`torch.Tensor`):
				The new key states to cache.
			value_states (`torch.Tensor`):
				The new value states to cache.
			layer_idx (`int`):
				The index of the layer to cache the states for.
			cache_kwargs (`Dict[str, Any]`, `optional`):
				Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

		Return:
			A tuple containing the updated key and value states.
		"""
		# Update the number of seen tokens
		if layer_idx == 0:
			self._seen_tokens += key_states.shape[-2]
		if cache_kwargs is not None and layer_idx not in [0,1]:
			local_size = cache_kwargs['local_size']
		# Update the cache
		if len(self.key_cache) <= layer_idx:
			self.query_cache.append(query_states)
			self.key_cache.append(key_states)
			self.value_cache.append(value_states)
			self.group_key.append(group_key_states)
			self.group_value.append(group_value_states)
			# if group_key_states is not None:
			# 	self.group_key.append(group_key_states) 
			# if group_value_states is not None:
			# 	self.group_value.append(group_value_states)
		else:
			self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
			self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
			if cache_kwargs is not None and layer_idx not in [0,1]:
				self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
				if local_size>0 and  self.key_cache[layer_idx].size(-2)>local_size:
					self.query_cache[layer_idx] = self.query_cache[layer_idx][...,-local_size:, :]
					self.key_cache[layer_idx] = self.key_cache[layer_idx][...,-local_size:, :]
					self.value_cache[layer_idx] = self.value_cache[layer_idx][...,-local_size:, :]
			
			if group_key_states is not None:
				self.group_key[layer_idx] = torch.cat([self.group_key[layer_idx], group_key_states], dim=-2)
			if group_value_states is not None:
				self.group_value[layer_idx] = torch.cat([self.group_value[layer_idx], group_value_states], dim=-2)
		if layer_idx in [0,1]:
			return self.key_cache[layer_idx], self.value_cache[layer_idx]
		if len(self.group_key) > 0:
			return self.key_cache[layer_idx], self.value_cache[layer_idx], \
				self.group_key[layer_idx], self.group_value[layer_idx]
		else:
			return self.key_cache[layer_idx], self.value_cache[layer_idx], \
				None, None

	def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
		"""Returns the sequence length of the cached states. A layer index can be optionally passed."""
		if len(self.key_cache) <= layer_idx:
			return 0
		return self._seen_tokens

	def get_max_length(self) -> Optional[int]:
		"""Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
		return None

	# TODO: How to implement beam search 
	def reorder_cache(self, beam_idx: torch.LongTensor):
		"""Reorders the cache for beam search, given the selected beam indices."""
		for layer_idx in range(len(self.key_cache)):
			device = self.key_cache[layer_idx].device
			self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
			device = self.value_cache[layer_idx].device
			self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

	def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
		"""Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
		legacy_cache = ()
		for layer_idx in range(len(self)):
			if len(self.group_key) == 0:
				legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], 
								None, None),)
			else:
				legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], 
								self.group_key[layer_idx], self.group_value[layer_idx]),)
		return legacy_cache

	@classmethod
	def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
		"""Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
		cache = cls()
		if past_key_values is not None:
			for layer_idx in range(len(past_key_values)):
				key_states, value_states, group_key, group_value, hidden_states = past_key_values[layer_idx]
				cache.update(key_states, value_states, group_key, group_value, hidden_states, layer_idx)
		return cache
