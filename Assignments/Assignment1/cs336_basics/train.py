import torch
import torch.nn as nn
import numpy.typing as npt
import time
from pathlib import Path

from cs336_basics.utils import save_checkpoint, ExperimentLogger, plot_loss_curve
from cs336_basics.data import get_batch

Path("runs").mkdir(exist_ok=True)
run_dir = Path("runs")

def train(train_data: npt.NDArray,
          val_data: npt.NDArray,
          batch_size: int,
          context_length: int,
          device: str,
          iteration: int,
          model: nn.Module,
          loss_function, 
          optimizer: torch.optim.Optimizer,
          iteration_save: int,
          iteration_evaluate: int,
          ):
    print("====== START TRAINING =======")
    model = model.to(device)
    assert model.device == device, f"model should be trained on {device} instead of {model.device}"

    output_path = run_dir / time.strftime("%Y%m%d-%H%M%S")
    output_path.mkdir(exist_ok=True)

    checkpoints_path = output_path / "checkpoints"
    checkpoints_path.mkdir(exist_ok=True)

    logger = ExperimentLogger(output_path / "metrics.csv")

    val_x, val_y = get_batch(val_data, batch_size, context_length, device)

    for i in range(1, iteration + 1):
        model.train()
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = loss_function(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]

        if i % iteration_save == 0:
            save_checkpoint(model, optimizer, i, checkpoints_path / f"checkpoint{i}")
            print(f"iteration{i}: train loss={loss}, checkpoint is saved to {checkpoints_path / f"checkpoint{i}"}.")

        if i % iteration_evaluate == 0:
            val_loss = evaluate(val_x, val_y, model, loss_function)
            logger.log(
                step=i, 
                train_loss = loss, 
                val_loss = val_loss, 
                lr=lr,
            )

    save_checkpoint(model, optimizer, iteration, checkpoints_path / f"checkpoint{iteration}")
    plot_loss_curve(output_path / "metrics.csv", output_path / "loss_curve.png")
    print("======== TRAINING COMPLETE =======")

@torch.no_grad
def evaluate(val_x, val_y, model, loss_function):
    model.eval()
    logits = model(val_x)
    return loss_function(logits, val_y)

def main(train_data,
         val_data,
         batch_size,
         context_length,
         device,
         iteration,
         model,
         loss_function,
         optimizer,
         iteration_save,
         iteration_evaluate
         ):
    
    train(train_data, val_data, batch_size, context_length, device, iteration, model, loss_function, optimizer, iteration_save, iteration_evaluate)

if __name__ == "__main__":
    main()