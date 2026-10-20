import math
import torch
import torch.nn as nn
import os
from collections.abc import Iterable
import typing
import csv
import time

def learning_rate_schedule(t: int,
                           alpha_max: float,
                           alpha_min: float,
                           t_w: int,
                           t_c: int) -> float:
    assert t_w <= t_c, f"t_w = {t_w} is the lower bound, t_c = {t_c} is the upper bound"
    if t < t_w:
        return t / t_w * alpha_max
    if t >= t_w and t <= t_c:
        return alpha_min + 0.5 * (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (alpha_max - alpha_min)
    return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      M: float,
                      eps = 1e-6):
    sum = .0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.data
        sum += torch.sum(torch.square(grad))

    sum = sum.item()

    grad_norm = math.sqrt(sum)
    if grad_norm > M:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.mul_(M / (grad_norm + eps))

            
def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_weights = model.state_dict()
    optimizer_weights = optimizer.state_dict()
    torch.save((iteration, model_weights, optimizer_weights), out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    i, model_weights, optimizer_weights = torch.load(src)
    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_weights)
    return i

class ExperimentLogger:
    def __init__(self, path):
        self.start_time = time.perf_counter()
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.f,
            fieldnames=[
                "step", 
                "wall_time_s", 
                "train_loss", 
                "val_loss",
                "lr", 
            ],
        )
        self.writer.writeheader()
        self.f.flush()

    def log(self, *, step, train_loss, val_loss, lr):
        self.writer.writerow({
            "step": step,
            "wall_time_s": time.perf_counter() - self.start_time,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss.item()),
            "lr": float(lr),
        })
        self.f.flush()

    def close(self):
        self.f.close()


def plot_loss_curve(
    csv_path: str | os.PathLike,
    output_path: str | os.PathLike | None = None
) -> None:
    import matplotlib.pyplot as plt

    steps: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    x = steps

    plt.figure()
    plt.plot(x, train_losses, label="train loss")
    plt.plot(x, val_losses, label="val loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss curve")

    if output_path is not None:
        plt.savefig(output_path)
        print(f"loss_curve.png is saved to {output_path}.")
    else:
        plt.show()
