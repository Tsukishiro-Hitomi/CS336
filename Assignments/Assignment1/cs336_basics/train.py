import torch
import torch.nn as nn
import numpy.typing as npt
import typing
import os

from cs336_basics.utils import save_checkpoint
from cs336_basics.data import get_batch

def train(data: npt.NDArray,
          batch_size: int,
          context_length: int,
          device: str,
          iteration: int,
          model: nn.Module,
          loss_function, 
          optimizer: torch.optim.Optimizer,
          output_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
          iteration_save: int,
          ):
    print("====== START TRAINING =======")
    model = model.to(device)
    assert model.device == device, f"model should be trained on {device} instead of {model.device}"
    model.train()
    for i in range(1, iteration + 1):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = loss_function(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % iteration_save == 0:
            save_checkpoint(model, optimizer, i, output_path)
            print(f"iteration{i}: loss={loss}, checkpoint saved to {output_path} successfully.")

    save_checkpoint(model, optimizer, iteration, output_path)
    print("======== TRAINING COMPLETE =======")

def main(data,
         batch_size,
         context_length,
         device,
         iteration,
         model,
         loss_function,
         optimizer,
         output_path,
         iteration_save):
    train(data, batch_size, context_length, device, iteration, model, loss_function, optimizer, output_path, iteration_save)

if __name__ == "__main__":
    main()