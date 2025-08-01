import math
import torch
import triton
import triton.language as tl


__all__ = ["attention"]


def maybe_contiguous(x):
	# only when the inner most dimension is contiguous can LDGSTS be used
	# so inner-dimension contiguity is enforced.
	return x.contiguous() if x.stride(-1) != 1 else x


def rounded_multiple(a, b):
	return (a + b - 1) // b * b


# --------------------------- public API ---------------------------
class CCAFlashAttention(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx,
		q: torch.Tensor,
		k: torch.Tensor,
		v: torch.Tensor,
		causal: bool,
		sm_scale: float,
		pool_size: int,
		window_size: int
	):
		Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
		assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
		assert Dk in {16, 32, 64, 128}

		B, H, M, D = q.shape
		N = k.shape[2]
		Hk, Hv = k.shape[1], v.shape[1]
		assert Hk == Hv, "num of heads in k and v should be equal"
		assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
		num_groups = H // Hk

		if sm_scale is None:
			sm_scale = 1. / math.sqrt(D)

		# contiguity
		q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

		# to work around https://github.com/openai/triton/issues/2441
		device = torch.cuda.device_of(q)
		num_sms = torch.cuda.get_device_properties(device).multi_processor_count

		n_pools = N - M
		with torch.cuda.device(device):
			config = get_fwd_config(B, H, M, N, D, causal)

			BLOCK_M, BLOCK_N, num_stages, num_warps = config
			
			divisible_m = M % BLOCK_M == 0
			divisible_n_core = n_pools % BLOCK_N == 0
			divisible_n_noncore = (N-n_pools) % BLOCK_N == 0

			# consider using 3d grid to avoid div & rem
			grid = (triton.cdiv(M, BLOCK_M), H, B)
			o = torch.empty_like(q)
			L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
			_fwd_kernel[grid](
				q, k, v, sm_scale,
				pool_size, window_size, n_pools,
				L, o,
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				v.stride(0), v.stride(1), v.stride(2), v.stride(3),
				o.stride(0), o.stride(1), o.stride(2), o.stride(3),
				B, H, M, N, num_groups,
				BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
				IS_CAUSAL=causal,
				DIVISIBLE_M=divisible_m, 
				DIVISIBLE_N_CORE=divisible_n_core, DIVISIBLE_N_NONCORE=divisible_n_noncore,
				num_warps=num_warps, num_stages=num_stages,
			)

		# autograd context maintenance
		ctx.save_for_backward(q, k, v, o, L)
		ctx.sm_scale = sm_scale
		ctx.causal = causal
		ctx.pool_size = pool_size
		ctx.window_size = window_size
		ctx.n_pools = n_pools

		return o

	@staticmethod
	def backward(ctx, do, *ignored):
		q, k, v, o, L = ctx.saved_tensors
		sm_scale = ctx.sm_scale
		causal = ctx.causal

		B, H, M, D = q.shape
		N = k.shape[2]
		Hk = k.shape[1]
		num_groups = H // Hk

		if sm_scale is None:
			sm_scale = 1. / math.sqrt(D)

		# to work around https://github.com/openai/triton/issues/2441
		device = torch.cuda.device_of(q)
		with torch.cuda.device(device):
			config = get_bwd_config(B, H, M, N, D, causal)
			BLOCK_M, BLOCK_N, num_stages, num_warps = config
			# if ctx.n_pools % BLOCK_N > 0:
			# 	BLOCK_N=32
			assert ctx.n_pools % BLOCK_N == 0, f"The number of core tokens {ctx.n_pools} must be multiple of BLOCK_N {BLOCK_N}"
			# print(f'ctx.n_pools {ctx.n_pools} % BLOCK_N {BLOCK_N}')

			divisible_m = M % BLOCK_M == 0
			divisible_n_core = ctx.n_pools % BLOCK_N == 0
			divisible_n_noncore = (N-ctx.n_pools) % BLOCK_N == 0

			delta = torch.empty_like(L)
			grid = (triton.cdiv(M, BLOCK_M), H, B)
			_bwd_preprocess[grid](
				o, do,
				delta,
				o.stride(0), o.stride(1), o.stride(2), o.stride(3),
				do.stride(0), do.stride(1), do.stride(2), do.stride(3),
				delta.stride(0), delta.stride(1), delta.stride(2),
				M,
				BLOCK_M=BLOCK_M, D_HEAD=D,
				DIVISIBLE_M=divisible_m,
			)

			# NOTE that dk & dv always have the same number of heads as q, instead of q.
			dk = torch.empty((B, H, N, D), dtype=k.dtype, device=q.device)
			dv = torch.empty((B, H, N, D), dtype=v.dtype, device=q.device)
			grid = (triton.cdiv(N, BLOCK_N), H, B)
			_bwd_kv_kernel[grid](
				q, k, v, sm_scale, do,
				dk, dv,
				L, delta,
				ctx.pool_size, ctx.window_size, ctx.n_pools,
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				v.stride(0), v.stride(1), v.stride(2), v.stride(3),
				do.stride(0), do.stride(1), do.stride(2), do.stride(3),
				dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
				dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
				B, H, M, N,
				num_groups,
				BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
				CAUSAL=causal,
				DIVISIBLE_M=divisible_m, 
				DIVISIBLE_N_CORE=divisible_n_core, DIVISIBLE_N_NONCORE=divisible_n_noncore,
				num_stages=num_stages, num_warps=num_warps,
			)

			dq = torch.zeros_like(q)
			grid = (triton.cdiv(M, BLOCK_M), H, B)
			_bwd_q_kernel[grid](
				q, k, v, sm_scale, do,
				dq,
				L, delta,
				ctx.pool_size, ctx.window_size, ctx.n_pools,
				q.stride(0), q.stride(1), q.stride(2), q.stride(3),
				k.stride(0), k.stride(1), k.stride(2), k.stride(3),
				v.stride(0), v.stride(1), v.stride(2), v.stride(3),
				do.stride(0), do.stride(1), do.stride(2), do.stride(3),
				dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
				B, H, M, N,
				num_groups,
				BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
				CAUSAL=causal,
				DIVISIBLE_M=divisible_m, 
				DIVISIBLE_N_CORE=divisible_n_core, DIVISIBLE_N_NONCORE=divisible_n_noncore,
				num_stages=num_stages, num_warps=num_warps,
			)
			dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
			dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
		return dq, dk, dv, None, None, None, None, None, None


def cca_attention_v2(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	causal: bool = True,
	sm_scale: float = None,
	pool_size: int = 16,
	window_size: int = 1024
):
	# we would highlight that the current implementation has some wrong on input tensors with float32 dtype,
	# the reaons are still unclear. It works well on input tensors with float16 and bfloat16 dtypes
	return CCAFlashAttention.apply(q, k, v, causal, sm_scale, pool_size, window_size)


# --------------------------- Forward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_fwd_config(B, H, M, N, D, causal):
	if torch.cuda.get_device_capability() == (8, 0):
		if not causal:
			if D <= 64:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
			else:
				if M <= 1024:
					BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
				else:
					BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
		else:
			if D <= 64:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
			else:
				if M <= 1024:
					BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
				else:
					BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
					BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 8
	elif torch.cuda.get_device_capability() == (8, 6):
		if not causal:
			if D <= 64:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
			else:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
		else:  # causal
			if D <= 64:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
			else:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
	else:
		BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
	return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _core_fwd_kernel_inner(
	acc, l_i, m_i, q,
	K, V, qk_scale,
	start_m,
	offs_m, offs_n_base, offs_k,
	pool_size, window_size, n_pools,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	Z, H, M, N,
	BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
	IS_CAUSAL: tl.constexpr,
	DIVISIBLE_N: tl.constexpr,
	input_dtype,
):
	if IS_CAUSAL:
		hi = tl.minimum(n_pools, (start_m + 1) * BLOCK_M // pool_size)
	else:
		hi = n_pools

	# loop over k, v and update accumulators
	offs_n_init = offs_n_base
	k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn)  # (BLOCK_DMODEL, BLOCK_N)
	v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
	for start_n in range(0, hi, BLOCK_N):
		start_n = tl.multiple_of(start_n, BLOCK_N)
		offs_n = start_n + offs_n_base

		# -- load k, v --
		# if DIVISIBLE_N:
			# k = tl.load(k_ptrs, cache_modifier=".cg")
			# v = tl.load(v_ptrs, cache_modifier=".cg")
		# else:
		#     mask_n = offs_n < n_pools
		#     k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
		#     v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")		
		mask_n = offs_n < n_pools
		k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
		v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
		# -- compute qk ---
		s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		s += tl.dot(q, k)

		if not DIVISIBLE_N:
			s += tl.where(mask_n[None, :], 0, -1.0e7)
		if IS_CAUSAL:
			core_token_mask = ((offs_m[:, None]-(pool_size-1)-window_size)//pool_size) >= offs_n[None, :]
			s += tl.where(core_token_mask, 0, -1.0e7)

		m_i_new = tl.maximum(m_i, tl.max(s, 1))
		alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
		p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

		# -- compute partial sumexpn before applying dropout
		p_sum = tl.sum(p, 1)

		# -- scale and update acc: acc *= alpha[:, None]--
		acc *= alpha[:, None]
		acc += tl.dot(p.to(input_dtype), v)

		# -- update m_i and l_i --
		l_i = l_i * alpha + p_sum
		m_i = m_i_new
		# update pointers
		k_ptrs += BLOCK_N * stride_kn
		v_ptrs += BLOCK_N * stride_vn
	return acc, l_i, m_i
	
@triton.jit
def _noncore_fwd_kernel_inner(
	acc, l_i, m_i, q,
	K, V, qk_scale,
	start_m,
	offs_m, offs_n_base, offs_k,
	pool_size, window_size, n_pools,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	Z, H, M, N,
	BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
	IS_CAUSAL: tl.constexpr,
	DIVISIBLE_N: tl.constexpr,
	input_dtype,
):
	if IS_CAUSAL:
		hi = tl.minimum(N, (start_m + 1) * BLOCK_M)
		lo = tl.maximum(hi-window_size-pool_size-BLOCK_M, 0)
	else:
		lo = 0
		hi = N

	# loop over k, v and update accumulators
	offs_n_init = offs_n_base
	k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn)  # (BLOCK_DMODEL, BLOCK_N)
	v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
	k_ptrs += (lo+n_pools) * stride_kn
	v_ptrs += (lo+n_pools) * stride_vn

	for start_n in range(lo, hi, BLOCK_N):
		offs_n = start_n + offs_n_base

		# -- load k, v --
		# if DIVISIBLE_N:
		# 	k = tl.load(k_ptrs, cache_modifier=".cg")
		# 	v = tl.load(v_ptrs, cache_modifier=".cg")
		# else:
		# 	mask_n = offs_n+n_pools < N
		# 	k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
		# 	v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

		mask_n = offs_n+n_pools < N
		k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
		v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
		# -- compute qk ---
		s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		s += tl.dot(q, k)

		if not DIVISIBLE_N:
			s += tl.where(mask_n[None, :], 0, -1.0e6)
		if IS_CAUSAL:
			lower_triangle_mask = (offs_m[:, None]) >= (offs_n[None, :])
			s += tl.where(lower_triangle_mask, 0, -1.0e6)
			sliding_length = (offs_m+1) % pool_size + window_size
			sliding_mask = (offs_n[None, :]+1) > (offs_m[:, None]+1-sliding_length[:, None])
			s += tl.where(sliding_mask, 0, -1.0e6)

		# -- compute scaling constant ---
		m_i_new = tl.maximum(m_i, tl.max(s, 1))
		alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
		p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

		# -- compute partial sumexpn before applying dropout
		p_sum = tl.sum(p, 1)

		# -- scale and update acc: acc *= alpha[:, None]--
		acc *= alpha[:, None]
		acc += tl.dot(p.to(input_dtype), v)

		# -- update m_i and l_i --
		l_i = l_i * alpha + p_sum
		m_i = m_i_new
		# update pointers
		k_ptrs += BLOCK_N * stride_kn
		v_ptrs += BLOCK_N * stride_vn

	return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
	Q, K, V, sm_scale,
	pool_size, window_size, n_pools,
	L, O,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_oz, stride_oh, stride_om, stride_ok,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	IS_CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, 
	DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
):
	input_dtype = Q.dtype.element_ty
	# -- grid id --
	start_m = tl.program_id(0)
	off_h = tl.program_id(1)
	off_z = tl.program_id(2)

	log2e: tl.constexpr = 1.4426950408889634
	qk_scale = sm_scale * log2e

	# offset pointers for (batch, head)
	off_hk = off_h // num_groups
	Q += off_z * stride_qz + off_h * stride_qh
	K += off_z * stride_kz + off_hk * stride_kh
	V += off_z * stride_vz + off_hk * stride_vh
	O += off_z * stride_oz + off_h * stride_oh
	L += (off_z * H + off_h) * M  # l's shape is (B, H, M)

	offs_m_base = tl.arange(0, BLOCK_M)
	offs_m = start_m * BLOCK_M + offs_m_base
	offs_n_base = tl.arange(0, BLOCK_N)
	offs_k = tl.arange(0, BLOCK_DMODEL)

	# initialize pointers to value-like data
	q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)
	o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)  # (BLOCK_M, BLOCK_DMODEL)
	l_ptrs = L + offs_m

	# initialize pointer to m and l, fp32 for accumulators
	m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
	l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
	acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

	# load q
	# if DIVISIBLE_M:
	# 	q = tl.load(q_ptrs, cache_modifier=".cg")
	# else:
	# 	mask_m = offs_m < M
	# 	q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
	mask_m = offs_m < M
	q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

	# Dot I trick: to place q in registers, it saves shared memory
	if BLOCK_DMODEL < 128:
		I = tl.where(offs_k[:, None] == offs_k,
					 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
					 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
		q = tl.dot(q, I).to(input_dtype)

	# since attention mao must be not none for any raw in non-core token attention
	# we first run non-core token attention and then core token attention
	acc, l_i, m_i = _noncore_fwd_kernel_inner(
			acc, l_i, m_i, q,
			K, V, qk_scale,
			start_m,
			offs_m, offs_n_base, offs_k,
			pool_size, window_size, n_pools,
			stride_kz, stride_kh, stride_kn, stride_kk,
			stride_vz, stride_vh, stride_vn, stride_vk,
			Z, H, M, N,
			BLOCK_M, BLOCK_N,
			IS_CAUSAL,
			DIVISIBLE_N_NONCORE,
			input_dtype,
		)
	acc, l_i, m_i = _core_fwd_kernel_inner(
		acc, l_i, m_i, q,
		K, V, qk_scale,
		start_m,
		offs_m, offs_n_base, offs_k,
		pool_size, window_size, n_pools,
		stride_kz, stride_kh, stride_kn, stride_kk,
		stride_vz, stride_vh, stride_vn, stride_vk,
		Z, H, M, N,
		BLOCK_M, BLOCK_N,
		IS_CAUSAL,
		DIVISIBLE_N_CORE,
		input_dtype,
	)

	# write back l & o
	acc = acc * (1.0 / l_i[:, None])
	l = m_i * sm_scale + tl.log(l_i)  # log(normalizer)

	if DIVISIBLE_M:
		tl.store(l_ptrs, l, cache_modifier=".cg")
		tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
	else:
		tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
		tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


# --------------------------- Backward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_bwd_config(B, H, M, N, D, causal):
	if torch.cuda.get_device_capability() == (8, 0):
		if not causal:
			BLOCK_M = 128 if D <= 64 else 64
			BLOCK_N = 64
			num_stages = 2
			num_warps = 4
		else:
			BLOCK_M = 64
			BLOCK_N = 64
			num_stages = 3 if D <= 64 else 2
			num_warps = 4
	elif torch.cuda.get_device_capability() == (8, 6):  # tune for RTX-3090, device_capability(8, 6)
		if not causal:
			if D <= 64:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
			else:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
		else:
			if D <= 64:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
			else:
				BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
	else:
		BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
	return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _bwd_preprocess(
	Out, DO,
	Delta,
	stride_oz, stride_oh, stride_om, stride_ok,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dz, stride_dh, stride_dm,
	M,
	BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
	DIVISIBLE_M: tl.constexpr,
):
	off_h = tl.program_id(1)
	off_z = tl.program_id(2)
	Out += off_z * stride_oz + off_h * stride_oh
	DO += off_z * stride_doz + off_h * stride_doh
	Delta += off_z * stride_dz + off_h * stride_dh

	# compute (Out * Dout).sum() for vector interpretation
	off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
	off_n = tl.arange(0, D_HEAD)

	# load
	o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
	do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

	if DIVISIBLE_M:
		o = tl.load(o_ptrs).to(tl.float32)
		do = tl.load(do_ptrs).to(tl.float32)
	else:
		mask_m = off_m < M
		o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
		do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

	# mask_m = off_m < M
	# o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
	# do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

	# compute
	delta = tl.sum(o * do, axis=1)

	# (NOTE) dropout scaling doesn't affect delta's value
	# when dropout is applied, o and do are actually scaled.
	# original_o equals o times reverse scale while original_do is do times scale,
	# and thus delta remains unchanged.

	# write-back
	d_ptrs = Delta + off_m * stride_dm
	if DIVISIBLE_M:
		tl.store(d_ptrs, delta)
	else:
		tl.store(d_ptrs, delta, mask=mask_m)

@triton.jit
def _core_bwd_kv_kernel_inner(
	Q, K, V, sm_scale, DO,
	DK, DV,
	L,
	D,
	qk_scale, log2e,
	start_n,
	pool_size,
	window_size,
	n_pools,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dkz, stride_dkh, stride_dkn, stride_dkk,
	stride_dvz, stride_dvh, stride_dvn, stride_dvk,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
	input_dtype
):
	if CAUSAL:
		lo = tl.maximum(start_n * BLOCK_N * pool_size+window_size-1, 0)
		hi = M
	else:
		lo = 0
		hi = M

	offs_m_init = lo + tl.arange(0, BLOCK_M)
	offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_m_base = tl.arange(0, BLOCK_M)
	offs_k = tl.arange(0, BLOCK_DMODEL)

	# initialize pointers to value-like data
	q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)
	k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
	v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)
	do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok)  # (BLOCK_M, BLOCK_DMODEL)

	dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)  # (BLOCK_N, BLOCK_DMODEL)
	dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)  # (BLOCK_N, BLOCK_DMODEL)

	# k and v stay in SRAM throughout
	if DIVISIBLE_N_CORE:
		v = tl.load(v_ptrs)
		k = tl.load(k_ptrs)
	else:
		mask_n = offs_n < n_pools
		v = tl.load(v_ptrs, mask=mask_n[:, None])
		k = tl.load(k_ptrs, mask=mask_n[:, None])
	# mask_n = offs_n < n_pools
	# v = tl.load(v_ptrs, mask=mask_n[:, None])
	# k = tl.load(k_ptrs, mask=mask_n[:, None])

	# initialize dk amd dv
	dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
	dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

	# loop over a col
	for start_m in range(lo, hi, BLOCK_M):
		start_m = tl.multiple_of(start_m, BLOCK_M)
		offs_m = start_m + offs_m_base
		core_token_mask = ((offs_m[:, None]-(pool_size-1)-window_size)//pool_size) >= offs_n[None, :]

		# load q1, k1, q2, k2, v, do on-chip
		# if DIVISIBLE_M:
		# 	q = tl.load(q_ptrs)
		# else:
		# 	mask_m = offs_m < M
		# 	valid_mask = mask_m[:, None]  # & mask_n
		# 	q = tl.load(q_ptrs, mask=mask_m[:, None])
		mask_m = offs_m < M
		valid_mask = mask_m[:, None]  # & mask_n
		q = tl.load(q_ptrs, mask=mask_m[:, None])

		# recompute p = softmax(qk * sm_scale, dim=-1)
		s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		s += tl.dot(q, tl.trans(k))

		# -- recompute p ---
		# if DIVISIBLE_M:
		# 	l = tl.load(L + offs_m)
		# else:
		# 	l = tl.load(L + offs_m, mask=mask_m)
		l = tl.load(L + offs_m, mask=mask_m)

		p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

		if not DIVISIBLE_M:
			p = tl.where(valid_mask, p, 0.0)
		if CAUSAL:
			p = tl.where(core_token_mask, p, 0.0)

		# compute dv = dot(p, do)
		# if DIVISIBLE_M:
		# 	do = tl.load(do_ptrs)
		# else:
		# 	do = tl.load(do_ptrs, mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)
		do = tl.load(do_ptrs, mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)

		dv += tl.dot(tl.trans(p).to(input_dtype), do)  # (BLOCK_N, BLOCK_DMODEL)  # still correct

		# compute dp = dot(v, do)
		# if DIVISIBLE_M:
		# 	delta = tl.load(D + offs_m)
		# else:
		# 	delta = tl.load(D + offs_m, mask=mask_m)
		delta = tl.load(D + offs_m, mask=mask_m)

		dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		dp += tl.dot(do, tl.trans(v))

		# compute ds = p * (dp - delta[:, None])
		ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

		if not DIVISIBLE_M:
			ds = tl.where(valid_mask, ds, 0.0)
		if CAUSAL:
			ds = tl.where(core_token_mask, ds, 0.0)
		ds = ds.to(input_dtype)

		# compute dk = dot(ds.T, q) masking
		dk += tl.dot(tl.trans(ds), q)

		# increment pointers
		q_ptrs += BLOCK_M * stride_qm
		do_ptrs += BLOCK_M * stride_dom

	dk *= sm_scale
	if DIVISIBLE_N_CORE:
		tl.store(dk_ptrs, dk.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
		tl.store(dv_ptrs, dv.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL,)
	else:
		mask_n = offs_n < n_pools
		tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL)
		tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL,)
	
	return dk, dv


@triton.jit
def _noncore_bwd_kv_kernel_inner(
	Q, K, V, sm_scale, DO,
	DK, DV,
	L,
	D,
	qk_scale, log2e,
	start_n,
	pool_size,
	window_size,
	n_pools,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dkz, stride_dkh, stride_dkn, stride_dkk,
	stride_dvz, stride_dvh, stride_dvn, stride_dvk,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
	input_dtype
):
	if CAUSAL:
		lo = tl.maximum(start_n * BLOCK_N-n_pools, 0)
		hi = tl.minimum(lo+window_size+pool_size+BLOCK_M, M)
	else:
		lo = 0
		hi = M

	offs_m_init = lo + tl.arange(0, BLOCK_M)
	offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
	offs_m_base = tl.arange(0, BLOCK_M)
	offs_k = tl.arange(0, BLOCK_DMODEL)

	# initialize pointers to value-like data
	q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)
	k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
	v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)
	do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok)  # (BLOCK_M, BLOCK_DMODEL)

	dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)  # (BLOCK_N, BLOCK_DMODEL)
	dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)  # (BLOCK_N, BLOCK_DMODEL)

	# k and v stay in SRAM throughout
	if DIVISIBLE_N_NONCORE:
		v = tl.load(v_ptrs)
		k = tl.load(k_ptrs)
	else:
		mask_n = offs_n < N
		v = tl.load(v_ptrs, mask=mask_n[:, None])
		k = tl.load(k_ptrs, mask=mask_n[:, None])
	# mask_n = offs_n < N
	# v = tl.load(v_ptrs, mask=mask_n[:, None])
	# k = tl.load(k_ptrs, mask=mask_n[:, None])

	# initialize dk amd dv
	dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
	dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

	# loop over a col
	for start_m in range(lo, hi, BLOCK_M):
		start_m = tl.multiple_of(start_m, BLOCK_M)
		offs_m = start_m + offs_m_base
		causal_mask = (offs_m[:, None]) >= (offs_n[None, :]-n_pools)  # (BLOCK_M, BLOCK_N)
		sliding_length = (offs_m+1) % pool_size + window_size
		sliding_mask = (offs_n[None, :]-n_pools+1) > (offs_m[:, None]+1-sliding_length[:, None])

		# load q1, k1, q2, k2, v, do on-chip
		# if DIVISIBLE_M:
		# 	q = tl.load(q_ptrs)
		# else:
		# 	mask_m = offs_m < M
		# 	valid_mask = mask_m[:, None]  # & mask_n
		# 	q = tl.load(q_ptrs, mask=mask_m[:, None])
		mask_m = offs_m < M
		valid_mask = mask_m[:, None]  # & mask_n
		q = tl.load(q_ptrs, mask=mask_m[:, None])

		# recompute p = softmax(qk * sm_scale, dim=-1)
		s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		s += tl.dot(q, tl.trans(k))

		# -- recompute p ---
		# if DIVISIBLE_M:
		# 	l = tl.load(L + offs_m)
		# else:
		# 	l = tl.load(L + offs_m, mask=mask_m)
		l = tl.load(L + offs_m, mask=mask_m)

		p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

		if not DIVISIBLE_M:
			p = tl.where(valid_mask, p, 0.0)
		if CAUSAL:
			p = tl.where(causal_mask, p, 0.0)
			p = tl.where(sliding_mask, p, 0.0)

		# compute dv = dot(p, do)
		# if DIVISIBLE_M:
		# 	do = tl.load(do_ptrs)
		# else:
		# 	do = tl.load(do_ptrs, mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)
		do = tl.load(do_ptrs, mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)

		dv += tl.dot(tl.trans(p).to(input_dtype), do)  # (BLOCK_N, BLOCK_DMODEL)  # still correct

		# compute dp = dot(v, do)
		# if DIVISIBLE_M:
		# 	delta = tl.load(D + offs_m)
		# else:
		# 	delta = tl.load(D + offs_m, mask=mask_m)			
		delta = tl.load(D + offs_m, mask=mask_m)

		dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		dp += tl.dot(do, tl.trans(v))

		# compute ds = p * (dp - delta[:, None])
		ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

		if not DIVISIBLE_M:
			ds = tl.where(valid_mask, ds, 0.0)
		if CAUSAL:
			ds = tl.where(causal_mask, ds, 0.0)
			ds = tl.where(sliding_mask, ds, 0.0)
		ds = ds.to(input_dtype)

		# compute dk = dot(ds.T, q) masking
		dk += tl.dot(tl.trans(ds), q)

		# increment pointers
		q_ptrs += BLOCK_M * stride_qm
		do_ptrs += BLOCK_M * stride_dom

	dk *= sm_scale
	if DIVISIBLE_N_NONCORE:
		tl.store(dk_ptrs, dk.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
		tl.store(dv_ptrs, dv.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL,)
	else:
		mask_n = offs_n < N
		tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL)
		tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL,)
	
	# mask_n = offs_n < N
	# tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL)
	# tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL,)
	return dk, dv

@triton.jit
def _bwd_kv_kernel(
	Q, K, V, sm_scale, DO,
	DK, DV,
	L,
	D,
	pool_size,
	window_size,
	n_pools,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dkz, stride_dkh, stride_dkn, stride_dkk,
	stride_dvz, stride_dvh, stride_dvn, stride_dvk,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
):
	input_dtype = Q.dtype.element_ty
	# -- grid id --
	start_n = tl.program_id(0)
	off_h = tl.program_id(1)
	off_z = tl.program_id(2)
	log2e: tl.constexpr = 1.4426950408889634
	qk_scale = sm_scale * log2e

	# offset pointers for (batch, head)
	off_hk = off_h // num_groups
	Q += off_z * stride_qz + off_h * stride_qh
	K += off_z * stride_kz + off_hk * stride_kh
	V += off_z * stride_vz + off_hk * stride_vh
	DO += off_z * stride_doz + off_h * stride_doh

	# offset pointers for batch/head
	DK += off_z * stride_dkz + off_h * stride_dkh
	DV += off_z * stride_dvz + off_h * stride_dvh

	# offset pointers for batch/head
	D += (off_z * H + off_h) * M
	L += (off_z * H + off_h) * M

	NONCORE_TOKEN_BWD = start_n * BLOCK_N >= n_pools
	if NONCORE_TOKEN_BWD:
		_noncore_bwd_kv_kernel_inner(
			Q, K, V, sm_scale, DO,
			DK, DV,
			L,
			D,
			qk_scale, log2e,
			start_n,
			pool_size,
			window_size,
			n_pools,
			stride_qz, stride_qh, stride_qm, stride_qk,
			stride_kz, stride_kh, stride_kn, stride_kk,
			stride_vz, stride_vh, stride_vn, stride_vk,
			stride_doz, stride_doh, stride_dom, stride_dok,
			stride_dkz, stride_dkh, stride_dkn, stride_dkk,
			stride_dvz, stride_dvh, stride_dvn, stride_dvk,
			Z, H, M, N,
			num_groups,
			BLOCK_M, BLOCK_DMODEL, BLOCK_N,
			CAUSAL,
			DIVISIBLE_M, DIVISIBLE_N_CORE, DIVISIBLE_N_NONCORE,
			input_dtype
		)
	else:
		_core_bwd_kv_kernel_inner(
			Q, K, V, sm_scale, DO,
			DK, DV,
			L,
			D,
			qk_scale, log2e,
			start_n,
			pool_size,
			window_size,
			n_pools,
			stride_qz, stride_qh, stride_qm, stride_qk,
			stride_kz, stride_kh, stride_kn, stride_kk,
			stride_vz, stride_vh, stride_vn, stride_vk,
			stride_doz, stride_doh, stride_dom, stride_dok,
			stride_dkz, stride_dkh, stride_dkn, stride_dkk,
			stride_dvz, stride_dvh, stride_dvn, stride_dvk,
			Z, H, M, N,
			num_groups,
			BLOCK_M, BLOCK_DMODEL, BLOCK_N,
			CAUSAL,
			DIVISIBLE_M, DIVISIBLE_N_CORE, DIVISIBLE_N_NONCORE,
			input_dtype
		)
	

@triton.jit
def _core_bwd_q_kernel_inner(
	q, dq, K, V,
	do,
	delta,
	l,
	qk_scale,
	log2e,
	start_m,
	offs_m, offs_n_base, offs_k,
	pool_size,
	window_size,
	n_pools,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dqz, stride_dqh, stride_dqm, stride_dqk,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
	input_dtype
):
	if CAUSAL:
		hi = tl.minimum(N, (start_m + 1) * BLOCK_M // pool_size)
	else:
		hi = N
	offs_n_init = offs_n_base
	k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
	v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)

	# loop over a row
	for start_n in range(0, hi, BLOCK_N):
		offs_n = start_n + offs_n_base

		# load k1, k2, v on chip
		# if DIVISIBLE_N_CORE:
		# 	v = tl.load(v_ptrs)
		# 	k = tl.load(k_ptrs)
		# else:
		# 	mask_n = offs_n < n_pools
		# 	v = tl.load(v_ptrs, mask=mask_n[:, None])
		# 	k = tl.load(k_ptrs, mask=mask_n[:, None])
		mask_n = offs_n < n_pools
		v = tl.load(v_ptrs, mask=mask_n[:, None])
		k = tl.load(k_ptrs, mask=mask_n[:, None])

		# recompute p = softmax(qk * sm_scale, dim=-1)
		if not DIVISIBLE_N_CORE:
			valid_mask = mask_n  # & mask_m[:, None]
		if CAUSAL:
			core_token_mask = ((offs_m[:, None]-(pool_size-1)-window_size)//pool_size) >= offs_n[None, :]

		s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		s += tl.dot(q, tl.trans(k))

		p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

		# compute dp = dot(v, do)
		dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		dp += tl.dot(do.to(input_dtype), tl.trans(v))

		ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

		# mask ds to ensure no small values
		# if not DIVISIBLE_N_CORE:
		# 	ds = tl.where(valid_mask, ds, 0.0)
			
		ds = tl.where(mask_n, ds, 0.0)
		if CAUSAL:
			ds = tl.where(core_token_mask, ds, 0.0)
			# ds = tl.where(sliding_mask, ds, 0.0)

		dq += tl.dot(ds.to(input_dtype), k)

		# increment pointers
		k_ptrs += BLOCK_N * stride_kn
		v_ptrs += BLOCK_N * stride_vn

	return dq

@triton.jit
def _noncore_bwd_q_kernel_inner(
	q, dq, K, V,
	do,
	delta,
	l,
	qk_scale,
	log2e,
	start_m,
	offs_m, offs_n_base, offs_k,
	pool_size,
	window_size,
	n_pools,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dqz, stride_dqh, stride_dqm, stride_dqk,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
	input_dtype
):
	# loop over k, v and update accumulator
	if CAUSAL:
		hi = tl.minimum(N, (start_m + 1) * BLOCK_M)
		lo = tl.maximum(hi-window_size-pool_size-BLOCK_M, 0)
	else:
		lo = 0
		hi = N

	offs_n_init = offs_n_base
	k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
	v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)
	k_ptrs += (lo+n_pools) * stride_kn
	v_ptrs += (lo+n_pools) * stride_vn

	# loop over a row
	for start_n in range(lo, hi, BLOCK_N):
		offs_n = start_n + offs_n_base

		# load k1, k2, v on chip
		# if DIVISIBLE_N_CORE:
		# 	v = tl.load(v_ptrs)
		# 	k = tl.load(k_ptrs)
		# else:
		# 	mask_n = offs_n+n_pools < N
		# 	v = tl.load(v_ptrs, mask=mask_n[:, None])
		# 	k = tl.load(k_ptrs, mask=mask_n[:, None])			
		mask_n = offs_n+n_pools < N
		v = tl.load(v_ptrs, mask=mask_n[:, None])
		k = tl.load(k_ptrs, mask=mask_n[:, None])

		# recompute p = softmax(qk * sm_scale, dim=-1)
		if not DIVISIBLE_N_CORE:
			valid_mask = mask_n  # & mask_m[:, None]
		if CAUSAL:
			causal_mask = (offs_m[:, None]) >= (offs_n[None, :])
			# s += tl.where(lower_triangle_mask, 0, -1.0e6)
			sliding_length = (offs_m+1) % pool_size + window_size
			sliding_mask = (offs_n[None, :]+1) > (offs_m[:, None]+1-sliding_length[:, None])

		s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		s += tl.dot(q, tl.trans(k))

		p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)  # (BLOCK_M, BLOCK_N)

		# compute dp = dot(v, do)
		dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
		dp += tl.dot(do.to(input_dtype), tl.trans(v))

		ds = p * (dp - delta[:, None])  # (BLOCK_M, BLOCK_N)

		# mask ds to ensure no small values
		# if not DIVISIBLE_N_CORE:
		# 	ds = tl.where(valid_mask, ds, 0.0)

		ds = tl.where(mask_n, ds, 0.0)

		if CAUSAL:
			ds = tl.where(causal_mask, ds, 0.0)
			ds = tl.where(sliding_mask, ds, 0.0)

		dq += tl.dot(ds.to(input_dtype), k)

		# increment pointers
		k_ptrs += BLOCK_N * stride_kn
		v_ptrs += BLOCK_N * stride_vn
	return dq

@triton.jit
def _bwd_q_kernel(
	Q, K, V, sm_scale, DO,
	DQ,
	L,
	D,
	pool_size,
	window_size,
	n_pools,
	stride_qz, stride_qh, stride_qm, stride_qk,
	stride_kz, stride_kh, stride_kn, stride_kk,
	stride_vz, stride_vh, stride_vn, stride_vk,
	stride_doz, stride_doh, stride_dom, stride_dok,
	stride_dqz, stride_dqh, stride_dqm, stride_dqk,
	Z, H, M, N,
	num_groups,
	BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
	CAUSAL: tl.constexpr,
	DIVISIBLE_M: tl.constexpr, DIVISIBLE_N_CORE: tl.constexpr, DIVISIBLE_N_NONCORE: tl.constexpr,
):
	# DIVISIBLE_N = DIVISIBLE_N_CORE
	input_dtype = Q.dtype.element_ty
	# -- grid id --
	start_m = tl.program_id(0)
	off_h = tl.program_id(1)
	off_z = tl.program_id(2)

	log2e: tl.constexpr = 1.4426950408889634
	qk_scale = sm_scale * log2e

	# offset pointers for (batch, head)
	off_hk = off_h // num_groups
	Q += off_z * stride_qz + off_h * stride_qh
	K += off_z * stride_kz + off_hk * stride_kh
	V += off_z * stride_vz + off_hk * stride_vh
	DO += off_z * stride_doz + off_h * stride_doh
	D += (off_z * H + off_h) * M
	L += (off_z * H + off_h) * M

	# offset pointers for batch/head
	DQ += off_z * stride_dqz + off_h * stride_dqh

	offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n_base = tl.arange(0, BLOCK_N)
	offs_n_init = offs_n_base
	offs_k = tl.arange(0, BLOCK_DMODEL)

	# initialize pointers to value-like data
	q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)

	dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)  # (BLOCK_M, BLOCK_DMODEL)
	do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok)  # (BLOCK_M, BLOCK_DMODEL)

	# pointer to row-wise quantities in value-like data
	d_ptrs = D + offs_m
	l_ptrs = L + offs_m

	# load q: it will stay in SRAM throughout
	# if DIVISIBLE_M:
	# 	q = tl.load(q_ptrs)
	# 	do = tl.load(do_ptrs)
	# 	delta = tl.load(d_ptrs)
	# 	l = tl.load(l_ptrs)
	# else:
	# 	mask_m = offs_m < M
	# 	q = tl.load(q_ptrs, mask=mask_m[:, None])
	# 	do = tl.load(do_ptrs, mask=mask_m[:, None])
	# 	delta = tl.load(d_ptrs, mask=mask_m)
	# 	l = tl.load(l_ptrs, mask=mask_m)
	mask_m = offs_m < M
	q = tl.load(q_ptrs, mask=mask_m[:, None])
	do = tl.load(do_ptrs, mask=mask_m[:, None])
	delta = tl.load(d_ptrs, mask=mask_m)
	l = tl.load(l_ptrs, mask=mask_m)

	# initialize dq
	dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)


	dq = _noncore_bwd_q_kernel_inner(
		q, dq, K, V,
		do,
		delta,
		l,
		qk_scale,
		log2e,
		start_m,
		offs_m, offs_n_base, offs_k,
		pool_size,
		window_size,
		n_pools,
		stride_qz, stride_qh, stride_qm, stride_qk,
		stride_kz, stride_kh, stride_kn, stride_kk,
		stride_vz, stride_vh, stride_vn, stride_vk,
		stride_doz, stride_doh, stride_dom, stride_dok,
		stride_dqz, stride_dqh, stride_dqm, stride_dqk,
		Z, H, M, N,
		num_groups,
		BLOCK_M, BLOCK_DMODEL, BLOCK_N,
		CAUSAL,
		DIVISIBLE_M, DIVISIBLE_N_CORE, DIVISIBLE_N_NONCORE,
		input_dtype
	)
	dq = _core_bwd_q_kernel_inner(
		q, dq, K, V,
		do,
		delta,
		l,
		qk_scale,
		log2e,
		start_m,
		offs_m, offs_n_base, offs_k,
		pool_size,
		window_size,
		n_pools,
		stride_qz, stride_qh, stride_qm, stride_qk,
		stride_kz, stride_kh, stride_kn, stride_kk,
		stride_vz, stride_vh, stride_vn, stride_vk,
		stride_doz, stride_doh, stride_dom, stride_dok,
		stride_dqz, stride_dqh, stride_dqm, stride_dqk,
		Z, H, M, N,
		num_groups,
		BLOCK_M, BLOCK_DMODEL, BLOCK_N,
		CAUSAL,
		DIVISIBLE_M, DIVISIBLE_N_CORE, DIVISIBLE_N_NONCORE,
		input_dtype
	)

	dq *= sm_scale
	if DIVISIBLE_M:
		tl.store(dq_ptrs, dq.to(input_dtype))
	else:
		tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])
