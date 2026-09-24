import time
import numpy as np
from cs336_basics.tokenizer import tokenizer

owt_train_file_path = "data/owt_train.txt"
owt_train_out_path = "data/owt_train.bin"

owt_valid_file_path = "data/owt_valid.txt"
owt_valid_out_path = "data/owt_valid.bin"

owt_merges_path = "artifacts/tokenizer/owt_merges.pkl"
owt_vocab_path = "artifacts/tokenizer/owt_vocab.pkl"

tinystories_train_file_path = "data/TinyStoriesV2-GPT4-train.txt"
tinystories_train_out_path = "data/tinystories_train.bin"

tinystories_valid_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
tinystories_valid_out_path = "data/tinystories_valid.bin"

tinystories_merges_path = "artifacts/tokenizer/tinystories_merges.pkl"
tinystories_vocab_path = "artifacts/tokenizer/tinystories_vocab.pkl"

special_token = "<|endoftext|>"

SAMPLE_NUM = 10

def get_tokenizer(vocab_filepath, merges_filepath, special_tokens=None) -> tokenizer:
    return tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

def sample_first_n_files(n, file_path) -> list[str]:
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        current_doc = []

        for line in f:
            if "<|endoftext|>" in line:
                current_doc.append(line.split("<|endoftext")[0])
                doc = "".join(current_doc).strip()

                if doc:
                    documents.append(doc)
                if len(documents) == n:
                    break

                current_doc = []
            else:
                current_doc.append(line)
    return documents
        

def encode_experiment(samples, tokenizer_):
    total_bytes = sum(len(doc.encode("utf-8")) for doc in samples)
    start_time = time.perf_counter()
    encode_result = [tokenizer_.encode(doc) for doc in samples]
    encode_time = time.perf_counter() - start_time
    total_tokens = sum(len(r) for r in encode_result)
    compression_ratio = total_bytes / total_tokens
    encode_ratio = total_bytes / encode_time

    print(f"==== RESULT ====")
    print(f"total_bytes={total_bytes}")
    print(f"total_tokens={total_tokens}")
    print(f"compression ratio={compression_ratio:.2f} bytes/token")
    print(f"encode_time={encode_time:.2f}")
    print(f"encode_ratio={encode_ratio:.2f} bytes/second\n")

    return encode_result

def encode_dataset(input_path, output_path, tokenizer_):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "wb") as fout:

        current_doc = []

        for line in fin:
            if line.strip() == "<|endoftext|>":
                text = "".join(current_doc)
                current_doc = []

                if not text:
                    continue

                token_ids = tokenizer.encode(text)
                token_ids.append(
                    tokenizer.encode("<|endoftext|>")[0]
                )

                np.asarray(token_ids, dtype=np.uint16).tofile(fout)

        if current_doc:
            token_ids = tokenizer.encode("".join(current_doc))
            np.asarray(token_ids, dtype=np.uint16).tofile(fout)

def main():
    owt_tokenizer = get_tokenizer(owt_vocab_path, owt_merges_path, [special_token])
    owt_samples = sample_first_n_files(SAMPLE_NUM, owt_train_file_path)

    tinystories_tokenizer = get_tokenizer(tinystories_vocab_path, tinystories_merges_path, [special_token])
    tinystories_samples = sample_first_n_files(SAMPLE_NUM, tinystories_train_file_path)

    print("using owt_tokenizer to encode owt_samples:")
    encode_experiment(owt_samples, owt_tokenizer)

    print("using tinystories_tokenizer to encode tinystories_samples: \n")
    encode_experiment(tinystories_samples, tinystories_tokenizer)

    print("using tinystories_tokenizer to encode owt_samples: \n")
    encode_experiment(owt_samples, tinystories_tokenizer)

    encode_dataset(owt_train_file_path, owt_train_out_path, owt_tokenizer)
    print(f"{owt_train_file_path} encoding result saved to {owt_train_out_path}")

    encode_dataset(owt_valid_file_path, owt_valid_out_path, owt_tokenizer)
    print(f"{owt_valid_file_path} encoding result saved to {owt_valid_out_path} ")

    encode_dataset(tinystories_train_file_path, tinystories_train_out_path, tinystories_tokenizer)
    print(f"{tinystories_train_file_path} encoding result saved to {tinystories_train_out_path}")

    encode_dataset(tinystories_valid_file_path, tinystories_valid_out_path, tinystories_tokenizer)
    print(f"{tinystories_valid_file_path} encoding result saved to {tinystories_valid_out_path}")

if __name__ == "__main__":
    main()