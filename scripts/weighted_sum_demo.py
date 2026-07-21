from __future__ import annotations

import torch

from cs336_systems.weighted_sum import weighted_sum

def main() -> None:
    torch.manual_seed(0)

    x = torch.randn(32, 64, device="cuda", requires_grad=True)
    weight = torch.randn(64, device="cuda", requires_grad=True)
    grad_output = torch.randn(32, device="cuda")

    x_reference = x.detach().clone().requires_grad_(True)
    weight_reference = weight.detach().clone().requires_grad_(True)

    output = weighted_sum(x, weight)
    output_reference = (x_reference * weight_reference).sum(dim=-1)

    output.backward(grad_output)
    output_reference.backward(grad_output)
    torch.cuda.synchronize()

    print(output)
    print(f"device: {output.device}")
    print(f"grad_fn: {output.grad_fn}")
    print(f"forward max error: {(output - output_reference).abs().max().item():.8f}")
    print(f"grad_x max error: {(x.grad - x_reference.grad).abs().max().item():.8f}")
    print(f"grad_weight max error: {(weight.grad - weight_reference.grad).abs().max().item():.8f}")


if __name__ == "__main__":
    main()