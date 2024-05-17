from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from cs336_data.common_crawl import sample_warc_records
import fasttext
import re
# from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
from fastwarc import ArchiveIterator
import gzip
import os
import nltk
import hashlib
import mmh3
import unicodedata
import random
nltk.download('punkt')

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"


EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_REGEX = r'(?<!\d)(\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}(?!\d)'
IP_REGEX = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'

WARC_WET_FILE_PATH = "/home/shared/CC-examples/example.warc.wet.gz"
WARC_FILE_PATH = "/home/shared/CC-examples/example.warc.gz"

language_model_path = '/home/shared/lid.176.bin'
nsfw_model_path = '/home/shared/dolma-jigsaw-fasttext-bigrams-nsfw.bin'
toxic_model_path = '/home/shared/dolma-jigsaw-fasttext-bigrams-hatespeech.bin'
quality_model_path = "/home/c-mattreed/language-model-data/cs336-data/cs336_data/classifier/model.bin"

language_model = fasttext.load_model(language_model_path)
nsfw_model = fasttext.load_model(nsfw_model_path)
toxic_model = fasttext.load_model(toxic_model_path)
quality_model = fasttext.load_model(quality_model_path)

def clamp(num):
    return num
    return float(max(0, min(num, 1)))

def extract_text_from_html_bytes(html: bytes) -> str:
    encoding = detect_encoding(html)
    html_string = html.decode(encoding, errors="ignore")
    return extract_plain_text(html_string)

def identify_language(text: str):
    predictions = language_model.predict(text.replace("\n", " "))
    language_code = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return language_code, clamp(confidence)

def detect_nsfw(text: str):
    predictions = nsfw_model.predict(text.replace("\n", " "))
    nsfw_code = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return nsfw_code, clamp(confidence)

def detect_toxic(text: str):
    predictions = toxic_model.predict(text.replace("\n", " "))
    toxic_code = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return toxic_code, clamp(confidence)

def mask_emails(text: str):
    masked_text, count = re.subn(EMAIL_REGEX, EMAIL_MASK, text)
    return masked_text, count

def mask_phone_numbers(text: str):
    masked_text, count = re.subn(PHONE_REGEX, PHONE_MASK, text)
    return masked_text, count

def mask_ip_addresses(text: str):
    masked_text, count = re.subn(IP_REGEX, IP_MASK, text)
    return masked_text, count


def is_gopher_quality(text: str):
    words = nltk.word_tokenize(text)
    len_words = len(words)
    avg_word_len = sum(len(word) for word in words) / len_words if len_words else 0
    lines = nltk.line_tokenize(text)
    num_lines = len(lines)
    proportion_end_ellipsis = sum(1 for line in lines if line.endswith("...")) / num_lines if num_lines else 1
    proportion_alphabetic = sum(1 for word in words if any(char.isalpha() for char in word)) / len_words if len_words else 0

    num_words = len_words >= 50 and len_words <= 100000
    mean_word_len = avg_word_len >= 3 and avg_word_len <= 10
    ellipsis = proportion_end_ellipsis < 0.3
    alphabetic = proportion_alphabetic > 0.8

    # return num_words, mean_word_len, ellipsis, alphabetic
    return num_words and mean_word_len and ellipsis and alphabetic

def detect_quality(text: str):
    predictions = quality_model.predict(text.replace("\n", " "))
    quality_code = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]
    return quality_code, clamp(confidence)

def hash_line(line):
    return hashlib.md5(line.encode('utf-8')).hexdigest()

def exact_dedup(filepaths, output_dir):
    line_count = {}

    for file_path in filepaths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_hash = hash_line(line)
                if line_hash in line_count:
                    line_count[line_hash] += 1
                else:
                    line_count[line_hash] = 1

    for file_path in filepaths:
        unique_lines = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_hash = hash_line(line)
                if line_count[line_hash] == 1:
                    unique_lines.append(line)

        file_name = os.path.basename(file_path)
        new_file_path = os.path.join(output_dir, file_name)

        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.writelines(unique_lines)

def get_ngrams(text, n):
    words = nltk.word_tokenize(text)
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

def hash_ngram(ngram, seed):
    return mmh3.hash(ngram, seed=seed)

def normalize_text(text):
    text = text.lower() # Lowercase the text
    text = unicodedata.normalize('NFD', text) # Normalize unicode characters to NFD (Normalization Form Decomposition)
    text = ''.join([char for char in text if not unicodedata.combining(char)]) # Remove accents (combining characters)
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespaces
    
    return text

def estimate_jaccard(signature1, signature2):
    matches = sum(1 for i in range(len(signature1)) if signature1[i] == signature2[i])
    return matches / len(signature1)

def merge_clusters(clusters):
    merged = []
    for new_set in clusters:
        found = False
        for idx, existing_set in enumerate(merged):
            if not set(new_set).isdisjoint(existing_set):  # Check if sets overlap
                merged[idx] = existing_set.union(new_set)
                found = True
                break
        if not found:
            merged.append(set(new_set))
    return merged

def minhash_dedup(
        input_files: list[os.PathLike],
        num_hashes: int,
        num_bands: int,
        ngrams: int,
        jaccard_threshold: float,
        output_directory: os.PathLike
        ):
    signatures = {}

    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = normalize_text(file.read())
            text_ngrams = get_ngrams(text, ngrams)
            # print(text_ngrams)
            signature = [min(hash_ngram(ngram, i) for ngram in text_ngrams) for i in range(num_hashes)]
            signatures[file_path] = signature
    
    bucket_size = num_hashes // num_bands
    clusters = []
    for bucket in range(num_bands):
        bucket_signatures = {}
        for file_path, signature in signatures.items():
            band = tuple(signature[bucket * bucket_size: (bucket + 1) * bucket_size])
            if band in bucket_signatures:
                bucket_signatures[band].append(file_path)
            else:
                bucket_signatures[band] = [file_path]
        for filepaths in bucket_signatures.values():
            if len(filepaths) > 1:
                for i, a in enumerate(filepaths):
                    for j, b in enumerate(filepaths[i+1:]):
                        sig_a = signatures[a]
                        sig_b = signatures[b]
                        if estimate_jaccard(sig_a, sig_b) > jaccard_threshold:
                            clusters.append(set([a,b]))

    merged = merge_clusters(clusters)

    for cluster in merged:
        keep_file = random.choice(list(cluster))
        files_to_delete = cluster - {keep_file}
        for file_to_delete in files_to_delete:
            input_files.remove(file_to_delete)

    for file_path in input_files:

        with open(file_path, 'r', encoding='utf-8') as file_read:
            

            file_name = os.path.basename(file_path)
            new_file_path = os.path.join(output_directory, file_name)

            with open(new_file_path, 'w', encoding='utf-8') as file:
                file.write(file_read.read())

if __name__ == "__main__":
    pass
    # test_texts = [
    #     "This is a normal statement. that im interjecting with sex",
    #     "You are an idiot. maybe",
    #     "Check out this sexy photo of my cock",
    #     "Hello, how are you?",
    #     "FUCK YOU. I love you so much please love me"
    # ]
    # text = "I am writing a novel about the ways in which me like to have sex with eachother we are so sexy"
    # split = text.split(" ")
    # test_texts = []
    # for i in range(len(split)):
    #     test_texts.append(" ".join(split[:i]))

    # for text in test_texts:
    #     nsfw_label, nsfw_conf = detect_nsfw(text)
    #     toxic_label, toxic_conf = detect_toxic(text)
    #     print(f"Text: {text}")
    #     print(f"NSFW: {nsfw_label}, {nsfw_conf}")
    #     print(f"Toxic: {toxic_label}, {toxic_conf}")
    #     print('---')
    # text = open("test.txt", "r").read().encode("utf-8")
    # print(extract_text_from_html_bytes(text))
    # records = sample_warc_records(sample_size=20)
    # for i, record in enumerate(records):
    #     text = extract_text_from_html_bytes(record)
    #     language, confidence = identify_language(text)
    #     print(i, language, confidence, text[:100])

    with gzip.open(WARC_FILE_PATH, 'rb') as stream:
        # Create an ArchiveIterator
        archive_iterator = ArchiveIterator(stream)
        
        # Collect all records into a list
        total = 0
        toxic = 0
        nsfw = 0
        both = 0
        for record in archive_iterator:
            if record.headers.get('WARC-Type') == 'response':  # Consider only 'response' records
                content = record.reader.read()
                # original_text = extract_text_from_html_bytes(content)
                # text, num_emails = mask_emails(original_text)
                # text, num_phone_numbers = mask_phone_numbers(text)
                # text, num_ip_addresses = mask_ip_addresses(text)

                # if num_emails + num_phone_numbers + num_ip_addresses:
                #     tokens = [EMAIL_MASK, PHONE_MASK, IP_MASK]
                #     index = min([text.find(token) for token in tokens if token in text])
                #     print("FOUND: ", original_text[index:index+50])
                #     print("POUND: ", text[index:index+50])
                #     input()
                original_text = extract_text_from_html_bytes(content)
                # nsfw_label, nsfw_conf = detect_nsfw(original_text)
                # toxic_label, toxic_conf = detect_toxic(original_text)
                quality_label, quality_conf = detect_quality(original_text)


                total += 1

                if quality_label == "wiki":
                    print(original_text)
                    print(total, quality_label, quality_conf)
                    input()
                # if nsfw_label == "nsfw":
                #     nsfw += 1
                # if toxic_label == "toxic":
                #     toxic += 1

                # if nsfw_label == "nsfw" or toxic_label == "toxic":
                #     # both += 1
                #     print(original_text)
                #     print("NSFW: ", nsfw_label, nsfw_conf)
                #     print("TOXIC: ", toxic_label, toxic_conf)
                #     print(total)
                #     input()
                # num_words, mean_word_len, ellipsis, alphabetic = is_gopher_quality(original_text)

                # if (num_words and mean_word_len and ellipsis and alphabetic):
                #     print(original_text)
                #     print(total, num_words, mean_word_len, ellipsis, alphabetic)
                #     input()
                
                # print(total, nsfw, toxic, both)


