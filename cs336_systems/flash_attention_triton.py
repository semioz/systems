from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
) -> None:
    # Stride names are <tensor><axis>: b=batch, q=query position,
    # k=key/value position, and d=embedding dimension. There is no qk;
    # stride_kk is K's stride along its key-position axis.
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # This program owns one fixed Q tile, so begin at its query-tile offset.
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    q_i = tl.load(Q_block_ptr)

    # K starts at key tile 0. The single inner loop advances this pointer
    # through every K tile for the fixed Q tile above.
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V uses the same key-position tile as K, so it also starts at offset 0.
    # It advances beside K at the end of every inner-loop iteration.
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # O and L are the outputs for this program's fixed Q tile, so they use
    # the same query-tile offset as Q rather than starting at position 0.
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # for the current Q tile, unnormalized output accumulator
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    # running softmax denominator
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    # running row-wise maximum
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr)
        v_j = tl.load(V_block_ptr)

        scores = tl.dot(q_i, tl.trans(k_j)) * scale
        if is_causal:
            query_positions = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            key_positions = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = query_positions[:, None] >= key_positions[None, :]
            scores += tl.where(causal_mask, 0.0, -1e6)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p_tilde = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i = alpha * l_i + tl.sum(p_tilde, axis=1)
        o_i = tl.dot(p_tilde.to(v_j.dtype), v_j, acc=alpha[:, None] * o_i)
        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # after processing all KV tiles update
    o_i = o_i / l_i[:, None]
    l_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, l_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]

        assert k.shape == (batch_size, n_keys, d)
        assert v.shape == (batch_size, n_keys, d)
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"

        q_tile_size = 16
        k_tile_size = 16
        output = torch.empty_like(q)
        logsumexp = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            logsumexp.stride(0),
            logsumexp.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=d**-0.5,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=is_causal,
        )

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # Implement the compiled PyTorch recomputation backward in the next section.
        raise NotImplementedError
