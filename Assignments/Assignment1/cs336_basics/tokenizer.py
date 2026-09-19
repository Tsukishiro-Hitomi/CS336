# 训练 byte-level BPE tokenizer
import os
import regex as re
from typing import BinaryIO
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
VOCABULARY_SIZE = 10000
NUM_CHUNKS = 20

# 初始化 vocabulary
def _initialize_vocabulary(special_tokens: list[str]) -> list[bytes]:
    vocab = [bytes([i]) for i in range(256)]    # 将单元素序列传给 bytes
    for st in special_tokens:
        vocab.append(st.encode("utf-8"))
    return vocab

# 根据 split_special_token 对文件进行分割
def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # 只处理内部边界
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 对每个分块做 pre_tokenization 并统计
def _pre_tokenization_helper(chunked_text: str) -> dict[tuple[bytes, ...], int]:
    # 使用 GPT-2 风格的正则 PAT
    # 使用 finditer 而非 findall，用于避免储存巨大的 pre_tokenized_words
    counts = dict()
    pre_tokenized_words = re.finditer(PAT, chunked_text)
    for match in pre_tokenized_words:
        pre_tokenized_word = match.group()
        # encoded_bytes: list[int, ...]
        encoded_bytes = pre_tokenized_word.encode("utf-8")
        key = tuple(bytes([b]) for b in encoded_bytes)
        counts[key] = counts.get(key, 0) + 1
    return counts

# 整合所有分块的结果
def _pre_tokenization(file_path: str, 
                      desired_num_chunks: int, 
                      split_special_token: bytes) -> dict[tuple[bytes, ...], int]:
    with open(file_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, desired_num_chunks, split_special_token)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        counts = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            counts.append(_pre_tokenization_helper(chunk))

        merged_counts = sum((Counter(c) for c in counts), Counter())
        return dict(merged_counts)
    
# using loops: a naive version
def bpe_train(counts: dict[tuple[bytes, ...]],
              vocab: list[bytes]) -> tuple[list[bytes], list[tuple[bytes, bytes]]]:
    merged = []
    while len(vocab) < VOCABULARY_SIZE:
        frequency = dict()
        for key, value in counts.items():
            if len(key) < 2:
                pass
            for i in range(len(key) - 1):
                pre = key[i]
                cur = key[i + 1]
                frequency[(pre, cur)] = frequency.get((pre, cur), 0) + value
        max_freq = max(frequency.values())
        max_key_list = [k for k, v in frequency.items() if v == max_freq]
        max_freq_key = sorted(max_key_list)[-1]

        if max_freq_key is None:
            break

        merged_key = b''.join(max_freq_key)
        print(f"merged_key: {merged_key}")
        vocab.append(merged_key)
        merged.append(max_freq_key)
        print(f"vocab: {vocab}")

        # update the counts
        new_counts = dict()
        for key, value in counts.items():
            if len(key) < 2:
                new_counts[key] = value
                continue

            new_key = []
            for i in range(len(key) - 1):
                pre = key[i]
                cur = key[i + 1]
                if (pre, cur) in merged:
                    new_key.append(pre + cur)
                    i += 1
                else:
                    new_key.append(pre)
            new_counts[tuple(new_key)] = value
        counts = new_counts

    return vocab, merged_key

def main():
    file_path = "test.txt"
    counts = _pre_tokenization(file_path, 1, b"<|endoftext|>")
    vocab = _initialize_vocabulary(["<|endoftext|>"])
    bpe_train(counts, vocab)

if __name__ == "__main__":
    main()
