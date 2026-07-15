import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.dtype]]:
        dtypes = {"parameters": next(self.parameters()).dtype}

        x = self.fc1(x)
        dtypes["fc1 output"] = x.dtype

        x = self.relu(x)
        x = self.ln(x)
        dtypes["layer norm output"] = x.dtype

        x = self.fc2(x)
        dtypes["logits"] = x.dtype
        return x, dtypes


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script is meant to be run on a CUDA GPU.")

    device = "cuda"
    model = ToyModel(4, 3).to(device)
    inputs = torch.randn(8, 4, device=device)

    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, dtypes = model(inputs)
        loss = logits.mean()
        dtypes["loss"] = loss.dtype

    loss.backward()
    dtypes["gradients"] = next(model.parameters()).grad.dtype

    for name, dtype in dtypes.items():
        print(f"{name}: {dtype}")


if __name__ == "__main__":
    main()
