import gzip
import random
import argparse
import os
from fastwarc import ArchiveIterator
from cs336_data.processing import extract_text_from_html_bytes, is_gopher_quality, identify_language
from tqdm import tqdm
import fasttext

WARC_DIR = '/home/shared/CC-MAIN-2023-50-warc-filtered'

def sample_urls(file_path, sample_size):
    """
    Randomly sample URLs from a gzip-compressed file using reservoir sampling.
    
    :param file_path: Path to the gzip file containing URLs.
    :param sample_size: Number of URLs to sample.
    :return: A list of sampled URLs.
    """
    sampled_urls = []
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            url = line.strip()
            if i < sample_size:
                sampled_urls.append(url)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sampled_urls[j] = url
            if i % 10000 == 0:
                progress = i / 43580000 * 100
                print(f"Progress: {progress:.4f}%")
    return sampled_urls

def get_docs_from_warc(warc_filepath, max_len=None, quality_filter=False):
    docs = []
    num_rejects = 0
    with gzip.open(warc_filepath, 'rb') as stream:
        archive_iterator = ArchiveIterator(stream)
        for record in archive_iterator:
            if max_len and len(docs) >= max_len:
                break

            if record.headers.get('WARC-Type') == 'response':  # Consider only 'response' records
                content = record.reader.read()
                text = extract_text_from_html_bytes(content)
                if not quality_filter:
                    docs.append(text)
                else:
                    if is_gopher_quality(text) and identify_language(text)[0] == "en":
                        docs.append(text)
                    else:
                        num_rejects += 1
    print("Num Rejects: ", num_rejects)
    return docs


def create_fasttext_dataset(pos_warc_filepaths, neg_warc_filepaths, pos_label, neg_label, train_filepath, valid_filepath):
    pos_docs = []
    for filepath in pos_warc_filepaths:
        pos_docs += get_docs_from_warc(filepath, quality_filter=True)

    neg_docs = []
    for filepath in neg_warc_filepaths:
        neg_docs += get_docs_from_warc(filepath, quality_filter=True)
        if len(neg_docs) >= len(pos_docs):
            break

    neg_docs = neg_docs[:len(pos_docs)]

    assert len(pos_docs) == len(neg_docs)

    split = len(pos_docs) * 3 // 4
    pos_train = pos_docs[:split]
    neg_train = neg_docs[:split]
    pos_valid = pos_docs[split:]
    neg_valid = neg_docs[split:]

    with open(train_filepath, "w") as f:
        for pos, neg in tqdm(zip(pos_train, neg_train)):
            pos_text = pos.replace("\n", " ")
            neg_text = neg.replace("\n", " ")
            f.write(f"{pos_label} {pos_text}\n")
            f.write(f"{neg_label} {neg_text}\n")
    
    with open(valid_filepath, "w") as f:
        for pos, neg in tqdm(zip(pos_valid, neg_valid)):
            pos_text = pos.replace("\n", " ")
            neg_text = neg.replace("\n", " ")
            f.write(f"{pos_label} {pos_text}\n")
            f.write(f"{neg_label} {neg_text}\n")

def train_model(train_filepath, valid_filepath, save_path):
    model = fasttext.train_supervised(input=train_filepath, epoch=25, lr=1.0, wordNgrams=4, verbose=2, minCount=1)
    model.save_model(save_path)
    result = model.test(path=valid_filepath)
    print(f"Number of examples: {result[0]}")
    print(f"Precision at 1: {result[1]}")
    print(f"Recall at 1: {result[2]}")

def get_warc_files(directory):
    warc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.warc.filtered.gz'):
                warc_files.append(os.path.join(root, file))
    return warc_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='For creating data and training quality classifier')
    parser.add_argument('--choose_urls', action="store_true", help='Subsamples Wikipedia URLs')
    parser.add_argument('--create_dataset', action="store_true", help='currate fasttext dataset')
    parser.add_argument('--train_model', action="store_true", help='train fasttext dataset')
    parser.add_argument('--samples', type=int, help='Number of Samples')
    parser.add_argument('--num_files', type=int, default=1, help='Number of Files')
    parser.add_argument('--pos_warc_filepath', type=str, default="subsampled_positive_urls.warc.gz", help='pos examples')
    parser.add_argument('--neg_warc_filepath', type=str, default="/home/shared/CC-examples/example.warc.gz", help='pos examples')


    args = parser.parse_args()
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    if args.choose_urls:
        num_samples = args.samples
        samples = sample_urls("/home/shared/enwiki-20240420-extracted_urls.txt.gz", num_samples)
        part_size = len(samples) // args.num_files
        sublists = [samples[i * part_size:(i + 1) * part_size] for i in range(args.num_files)]
        for i in range(args.num_files):
            num = str(i)
            data_path = os.path.join(curr_dir, f'classifier/{num_samples}_urls_{num}.txt')
            with open(data_path, "w") as f:
                f.write("\n".join(sublists[i]))
    elif args.create_dataset:
        train_path = os.path.join(curr_dir, f'classifier/data.train')
        valid_path = os.path.join(curr_dir, f'classifier/data.valid')

        pos_filepaths = get_warc_files("/home/c-mattreed/language-model-data/warcs")
        neg_filepaths = get_warc_files(WARC_DIR)

        create_fasttext_dataset(pos_filepaths, neg_filepaths, "__label__wiki", "__label__cc", train_path, valid_path)
    elif args.train_model:
        train_path = os.path.join(curr_dir, f'classifier/data.train')
        valid_path = os.path.join(curr_dir, f'classifier/data.valid')
        save_path = os.path.join(curr_dir, "classifier/model.bin")
        train_model(train_path, valid_path, save_path)
