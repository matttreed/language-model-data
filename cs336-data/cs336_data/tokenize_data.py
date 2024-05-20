import numpy as np
# from transformers import GPT2Tokenizer
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import os

def tokenize_and_serialize(input_path, output_path, chunk_size=1024*1024):
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    print("Tokenizer loaded.")

    total_size = os.path.getsize(input_path)
    total_chunks = (total_size // chunk_size) + (1 if total_size % chunk_size else 0)

    print(f"Tokenizing and serializing the dataset into {output_path}...")
    print(f"Total number of chunks: {total_chunks}")

    total_tokens = 0

    with open(input_path, 'r', encoding='utf-8') as file:
        with open(output_path, 'wb') as output:
            progress_bar = tqdm(total=total_chunks, unit='chunk')
            while True:
                text = file.read(chunk_size)
                if not text:
                    break

                tokens = tokenizer.encode(text, add_special_tokens=True)
                total_tokens += len(tokens)

                np.array(tokens, dtype=np.uint16).tofile(output)

                progress_bar.update(1)
            progress_bar.close()

    # data = np.memmap(output_path, dtype=np.uint16, mode="r")

    # with open(input_path, 'r', encoding='utf-8') as file:
    #     text = file.read()
    #     assert text == tokenizer.decode(data)
    #     print(text)

    return total_tokens

if __name__ == "__main__":
    input_path = 'filtered_data/raw.txt'
    output_path = 'filtered_data/data_tokenized.bin'
    num_tokens = tokenize_and_serialize(input_path, output_path)
    print(f"Number of tokens in the dataset: {num_tokens}")