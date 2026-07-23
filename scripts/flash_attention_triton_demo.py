from __future__ import annotations

import torch

from cs336_systems.flash_attention_triton import FlashAttentionTriton


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if is_causal:
        sequence_length = q.shape[-2]
        positions = torch.arange(sequence_length, device=q.device)
        scores = scores.masked_fill(positions[None, :, None] < positions[None, None, :], -1e6)
    return torch.softmax(scores, dim=-1) @ v, torch.logsumexp(scores, dim=-1)


def main() -> None:
    torch.manual_seed(0)
    for is_causal in (False, True):
        q = torch.randn(4, 128, 64, device="cuda", requires_grad=True)
        k = torch.randn(4, 128, 64, device="cuda", requires_grad=True)
        v = torch.randn(4, 128, 64, device="cuda", requires_grad=True)

        output = FlashAttentionTriton.apply(q, k, v, is_causal)
        logsumexp = [tensor for tensor in output.grad_fn.saved_tensors if tensor.shape == (4, 128)][0]
        output_ref, logsumexp_ref = reference_attention(q, k, v, is_causal)

        torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(logsumexp, logsumexp_ref, rtol=1e-2, atol=1e-2)
        print(f"is_causal={is_causal}: passed", flush=True)


if __name__ == "__main__":
    main()
