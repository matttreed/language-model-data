from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from cs336_data.common_crawl import sample_warc_records
import fasttext
import re
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
from fastwarc import ArchiveIterator
import gzip

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"


EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_REGEX = r'(?<!\d)(\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}(?!\d)'
IP_REGEX = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'

WARC_WET_FILE_PATH = "/home/shared/CC-examples/example.warc.wet.gz"
WARC_FILE_PATH = "/home/shared/CC-examples/example.warc.gz"

language_model = fasttext.load_model('/home/shared/lid.176.bin')
nsfw_model = fasttext.load_model('/home/shared/dolma-jigsaw-fasttext-bigrams-nsfw.bin')
toxic_model = fasttext.load_model('/home/shared/dolma-jigsaw-fasttext-bigrams-hatespeech.bin')

def clamp(num):
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
    

if __name__ == "__main__":
    text = "hey elrkgmelrkgme lglerkm gelmgmk er fuck fuck me"
    print(detect_nsfw(text))
    print(detect_toxic(text))
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
                nsfw_label, nsfw_conf = detect_nsfw(original_text)
                toxic_label, toxic_conf = detect_toxic(original_text)

                print(nsfw_conf, toxic_conf)
                input()

                if nsfw_label == "toxic" or toxic_label == "toxic":
                    print(original_text)
                    print("NSFW: ", nsfw_label, nsfw_conf)
                    print("TOXIC: ", toxic_label, toxic_conf)
                    input()


