import time
import pickle
import threading
import psutil
import os
from pathlib import Path

from cs336_basics.tokenizer import run_bpe_train


TS_DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
OWT_DATA_PATH = "data/owt_train.txt"
OUTPUT_DIR = Path("artifacts/tokenizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    @property
    def peak_gb(self):
        return self.peak_rss / (1024 ** 3)


def main():
    print("====== START TRAINING ======\n")

    memory_monitor = MemoryMonitor()
    memory_monitor.start()


    vocab, merges = run_bpe_train(
        # input_path=TS_DATA_PATH,
        input_path=OWT_DATA_PATH,
        # vocab_size=10000,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    memory_monitor.stop()


    # with open(OUTPUT_DIR / "tinystories_vocab.pkl", "wb") as f:
        # pickle.dump(vocab, f)

    # with open(OUTPUT_DIR / "tinystories_merges.pkl", "wb") as f:
        # pickle.dump(merges, f)

    with open(OUTPUT_DIR / "owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(OUTPUT_DIR / "owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    special_bytes = {b"<|endoftext|>"}

    normal_tokens = [
        token
        for token in vocab.values()
        if token not in special_bytes
    ]

    longest_token = max(normal_tokens, key=len)
    print("====== RESULTS ======\n")

    print(f"Peak memory usage:    {memory_monitor.peak_gb:.3f} GB\n")

    print(f"Vocab size:           {len(vocab)}")
    print(f"Number of merges:     {len(merges)}")
    print(f"Longest token:        {longest_token!r}")
    print(f"Longest token length: {len(longest_token)} bytes\n")

    # print(
        # f"Saved vocab to: "
        # f"{OUTPUT_DIR / 'tinystories_vocab.pkl'}"
    # )
    # print(
        # f"Saved merges to: "
        # f"{OUTPUT_DIR / 'tinystories_merges.pkl'}"
    # )

    print(
        f"Saved vocab to: "
        f"{OUTPUT_DIR / 'owt_vocab.pkl'}"
    )
    print(
        f"Saved merges to: "
        f"{OUTPUT_DIR / 'owt_merges.pkl'}"
    )


if __name__ == "__main__":
    main()