from cs336_basics.transformer import TransformerLM
from cs336_basics.loss import cross_entropy
from cs336_basics.attention import RoPE
from cs336_basics.optimizer import AdamW
from cs336_basics.train import train
import numpy as np

data_path = "data/test.bin"
data = np.memmap(data_path, dtype=np.uint16, mode="r")
batch_size = 2
context_length = 20
device = "cpu"
iteration = 10
d_model = 32
loss_function = cross_entropy
rope = RoPE(theta=2, d_k=16, max_seq_len=context_length, device=device)
model = TransformerLM(d_model=d_model, num_heads=2, d_ff=48, rope=rope, vocab_size=200, context_length=context_length, num_layers=3)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.99, betas=(0.9, 0.99), eps=1e-5)
output_path = "artifacts/test"
iteration_save = 5

train(data, batch_size, context_length, device, iteration, model, loss_function, optimizer, output_path, iteration_save)