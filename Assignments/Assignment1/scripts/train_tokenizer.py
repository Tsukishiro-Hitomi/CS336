import argparse
import time
import pickle
import threading
import psutil
import os
from pathlib import Path

from cs336_basics.tokenizer import run_bpe_train


# =========================
# Configuration
# =========================

OUTPUT_DIR = Path("artifacts/tokenizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CONFIG = {
    "tinystories": {
        "input_path": "data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10000,
        "output_prefix": "tinystories",
    },
    "owt": {
        "input_path": "data/owt_train.txt",
        "vocab_size": 32000,
        "output_prefix": "owt",
    },
}

class MemoryMonitor:
    def __init__(self, interval=0.05):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.peak_rss = 0
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            rss = self.process.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            time.sleep(self.interval)

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join()

    @property
    def peak_gb(self):
        return self.peak_rss / (1024 ** 3)

def main(dataset: str):
    config = DATASET_CONFIG[dataset]

    input_path = config["input_path"]
    vocab_size = config["vocab_size"]
    output_prefix = config["output_prefix"]

    vocab_path = OUTPUT_DIR / f"{output_prefix}_vocab.pkl"
    merges_path = OUTPUT_DIR / f"{output_prefix}_merges.pkl"

    print("====== START TRAINING ======\n")
    print(f"Dataset:     {dataset}")
    print(f"Input path:  {input_path}")
    print(f"Vocab size:  {vocab_size}\n")

    # Start monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()

    start_time = time.perf_counter()

    try:
        vocab, merges = run_bpe_train(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
    finally:
        memory_monitor.stop()

    elapsed_time = time.perf_counter() - start_time

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # =========================
    # Statistics
    # =========================

    special_bytes = {b"<|endoftext|>"}

    normal_tokens = [
        token
        for token in vocab.values()
        if token not in special_bytes
    ]

    longest_token = max(normal_tokens, key=len)

    print("\n====== RESULTS ======\n")

    print(f"Training time:        {elapsed_time:.2f} s")
    print(f"Peak memory usage:    {memory_monitor.peak_gb:.3f} GB")
    print(f"Vocab size:           {len(vocab)}")
    print(f"Number of merges:     {len(merges)}")
    print(f"Longest token:        {longest_token!r}")
    print(f"Longest token length: {len(longest_token)} bytes\n")

    print(f"Saved vocab to:       {vocab_path}")
    print(f"Saved merges to:      {merges_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer."
    )

    parser.add_argument(
        "dataset",
        choices=DATASET_CONFIG.keys(),
        help="Dataset to use for BPE training.",
    )

    args = parser.parse_args()

    main(args.dataset)