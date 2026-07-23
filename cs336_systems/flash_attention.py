from __future__ import annotations

import torch


def flash_attention_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    query_tile_size: int = 16,
    key_tile_size: int = 16,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Q has shape (batch_size, n_queries, d).
    # K and V have shape (batch_size, n_keys, d).
    batch_size, n_queries, d = q.shape
    n_keys = k.shape[1]
    # Split Q into ceil(n_queries / query_tile_size) query tiles of shape
    # (batch_size, query_tile_size, d).
    # Split K and V into ceil(n_keys / key_tile_size) matching key/value tiles
    # of shape (batch_size, key_tile_size, d).
    assert k.shape == (batch_size, n_keys, d)
    assert v.shape == (batch_size, n_keys, d)

    num_query_tiles = (n_queries + query_tile_size - 1) // query_tile_size
    num_key_tiles = (n_keys + key_tile_size - 1) // key_tile_size

    scale = d**-0.5
    output = torch.empty_like(q)
    logsumexp = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

    # Process each Q tile against every K/V tile.
    for query_tile_index in range(num_query_tiles):
        query_start = query_tile_index * query_tile_size
        query_end = min(query_start + query_tile_size, n_queries)
        q_i = q[:, query_start:query_end, :]

        # O_i is the unnormalized attention output accumulated so far.
        o_i = torch.zeros_like(q_i, dtype=torch.float32)

        # l_i is the running softmax denominator for every query row.
        l_i = torch.zeros(
            (batch_size, q_i.shape[1]),
            device=q.device,
            dtype=torch.float32,
        )
        # m_i is the running maximum score for every query row.
        m_i = torch.full(
            (batch_size, q_i.shape[1]),
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )

        for key_tile_index in range(num_key_tiles):
            key_start = key_tile_index * key_tile_size
            key_end = min(key_start + key_tile_size, n_keys)

            k_j = k[:, key_start:key_end, :]
            v_j = v[:, key_start:key_end, :]

            # S_i^(j): [batch, Bq, Bk]
            scores = (q_i.float() @ k_j.float().transpose(-2, -1)) * scale
            if is_causal:
                query_positions = torch.arange(query_start, query_end, device=q.device)[:, None]
                key_positions = torch.arange(key_start, key_end, device=q.device)[None, :]
                scores = scores.masked_fill(query_positions[None, :, :] < key_positions[None, :, :], -1e6)

            # New numerically stable maximum for each query row.
            m_new = torch.maximum(m_i, scores.max(dim=-1).values)

            # Unnormalized probabilities for only the current K/V tile.
            p_tilde = torch.exp(scores - m_new.unsqueeze(-1))

            # Rescale the previous state because its max may have changed.
            alpha = torch.exp(m_i - m_new)

            # update the running softmax denominator
            l_i = alpha * l_i + p_tilde.sum(dim=-1)

            # Update the unnormalized weighted-value sum.
            o_i = alpha.unsqueeze(-1) * o_i + p_tilde @ v_j.float()

            m_i = m_new

        o_i = o_i / l_i.unsqueeze(-1)
        l_i = m_i + torch.log(l_i)
        output[:, query_start:query_end, :] = o_i.to(output.dtype)
        logsumexp[:, query_start:query_end] = l_i

    return output, logsumexp


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Call the tiled pure-PyTorch forward implementation.
        # Save L, Q, K, V, and O for the FlashAttention backward pass.
        # Store is_causal on ctx for the later backward implementation.
        output, logsumexp = flash_attention_forward_pytorch(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # Part (a) only requires forward; implement the recomputation backward later.
        raise NotImplementedError
