# 训练 byte-level BPE tokenizer
import os
import regex as re
from typing import BinaryIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 对原始文本进行初次分块，便于 CPU 并行处理
NUM_CHUNKS = 8

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

# 对每个分块进行再分块，去掉所有 special_tokens
def _process_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    if len(special_tokens) == 0:
        return [chunk]
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern_str = "|".join(escaped_tokens)
    pattern = re.compile(pattern_str)
    return pattern.split(chunk)
    

# 对每个处理好的分块做 pre_tokenization 并统计
def _pre_tokenization_helper(chunked_tokens: str) -> dict[tuple[bytes, ...], int]:
    # 使用 GPT-2 风格的正则 PAT
    # 使用 finditer 而非 findall，用于避免储存巨大的 pre_tokenized_words
    counts = dict()
    pre_tokenized_words = re.finditer(PAT, chunked_tokens)
    for match in pre_tokenized_words:
        pre_tokenized_word = match.group()
        # encoded_bytes: list[int, ...]
        encoded_bytes = pre_tokenized_word.encode("utf-8")
        key = tuple(bytes([b]) for b in encoded_bytes)
        counts[key] = counts.get(key, 0) + 1
    return counts

# 由每个 CPU 进程执行
def _process_chunk_worker(
    file_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    # 每个进程自己打开文件
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk = chunk_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    # 换行标准化
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    chunk_counts = Counter()

    # 去掉 special token
    for chunked_tokens in _process_special_tokens(chunk,special_tokens):
        chunk_counts.update( _pre_tokenization_helper(chunked_tokens))

    return dict(chunk_counts)

def _pre_tokenization(
    file_path: str | os.PathLike,
    desired_num_chunks: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    # 主进程只负责找 chunk boundaries
    with open(file_path, "rb") as f:
        split_special_token = b"<|endoftext|>" if len(special_tokens) == 0 else special_tokens[0].encode("utf-8")
        boundaries = _find_chunk_boundaries(f, desired_num_chunks, split_special_token)

    tasks = [
        (
            file_path,
            start,
            end,
            special_tokens
        ) for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_process_chunk_worker, *task)
            for task in tasks
        ]

        merged_counts = Counter()
        for future in futures:
            merged_counts.update(future.result())

    return dict(merged_counts)

    
# using loops: a naive version
def bpe_train(counts: dict[tuple[bytes, ...], int],
              vocab: list[bytes],
              vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merged = []
    while len(vocab) < vocab_size:
        frequency = dict()
        for key, value in counts.items():
            for i in range(len(key) - 1):
                pre = key[i]
                cur = key[i + 1]
                frequency[(pre, cur)] = frequency.get((pre, cur), 0) + value

        if len(frequency) == 0:
            break

        # find the most-frequent key to merge
        max_freq = max(frequency.values())
        max_key_list = [k for k, v in frequency.items() if v == max_freq]
        max_freq_key = sorted(max_key_list)[-1]

        if max_freq_key is None:
            break

        merged_key = b''.join(max_freq_key)
        vocab.append(merged_key)
        merged.append(max_freq_key)

        # update the counts
        new_counts = dict()
        for key, value in counts.items():
            if len(key) < 2:
                new_counts[key] = value
                continue

            new_key = []
            i = 0
            while i < len(key):
                if i < len(key) - 1 and (key[i], key[i + 1]) == max_freq_key:
                    new_key.append(key[i] + key[i + 1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_counts[tuple(new_key)] = new_counts.get(tuple(new_key), 0) + value
        counts = new_counts
    vocab = dict(enumerate(vocab))
    return vocab, merged

def run_bpe_train(input_path: str | os.PathLike,
                  vocab_size: int,
                  special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _initialize_vocabulary(special_tokens)
    counts = _pre_tokenization(input_path, NUM_CHUNKS, special_tokens)
    return bpe_train(counts, vocab, vocab_size)


