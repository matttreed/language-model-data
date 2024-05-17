import gzip
import random
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
from fastwarc import ArchiveIterator
import concurrent.futures
import os
from cs336_data.processing import detect_quality, detect_toxic, detect_nsfw, extract_text_from_html_bytes, mask_emails, mask_ip_addresses, mask_phone_numbers
import logging
import logging.handlers
import multiprocessing
from queue import Queue

WARC_WET_FILE_PATH = "/home/shared/CC-examples/example.warc.wet.gz"
WARC_FILE_PATH = "/home/shared/CC-examples/example.warc.gz"
WARC_DIR = '/home/shared/CC-MAIN-2023-50-warc-filtered'
DATA_DIR = "/home/c-mattreed/language-model-data/filtered_data"

def setup_logging():
    log_queue = multiprocessing.Queue(-1)  # No limit on size
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    listener = logging.handlers.QueueListener(log_queue, logging.FileHandler('app.log'))
    listener.start()
    return listener

def sample_warc_records(file_path=WARC_FILE_PATH, sample_size=1):
    """
    Randomly sample records from a .warc.gz file.

    :param file_path: Path to the .warc.gz file
    :param sample_size: Number of records to sample
    :return: List of sampled records
    """
    # Open the .warc.gz file
    with gzip.open(file_path, 'rb') as stream:
        # Create an ArchiveIterator
        archive_iterator = ArchiveIterator(stream)
        
        # Collect all records into a list
        all_records = []
        for record in archive_iterator:
            if record.headers.get('WARC-Type') == 'response':  # Consider only 'response' records
                record.freeze()
                all_records.append(record)
        
        # Check if we have enough records to sample
        if len(all_records) < sample_size:
            raise ValueError("Not enough records to sample the desired amount")

        # Randomly sample from the collected records
        sampled_records = random.sample(all_records, sample_size)
        
        # Extract the content of the sampled records
        sampled_content = []
        for record in sampled_records:
            content = record.reader.read()
            if content:  # Check if content is not empty
                sampled_content.append(content)
            else:
                print(f"Empty content for record with headers: {record.headers}")

        return sampled_content
    
def get_warc_files(directory):
    warc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.warc.filtered.gz'):
                warc_files.append(os.path.join(root, file))
    return warc_files

def log(message):
    pid = multiprocessing.current_process().pid
    logging.info(f"{pid}: {message}")


def process_warc_file(filename):

    base_name = os.path.basename(filename).replace(".warc.filtered.gz", ".txt")
    new_file_path = os.path.join(DATA_DIR, base_name)

    log(f"Populating {base_name}")

    with open(new_file_path, "a") as output:

        with gzip.open(filename, 'rb') as stream:
            archive_iterator = ArchiveIterator(stream)
            
            for record in archive_iterator:
                if record.headers.get('WARC-Type') == 'response':
                    content = record.reader.read()
                    text = extract_text_from_html_bytes(content)

                    if detect_toxic(text)[0] == "toxic":
                        continue

                    if detect_quality(text)[0] == "cc":
                        continue

                    text, _ = mask_emails(text)
                    text, _ = mask_ip_addresses(text)
                    text, _ = mask_phone_numbers(text)

                    output.write(text)
                    output.write("\n<|endoftext|>\n")




def main():
    warc_files = get_warc_files(WARC_DIR)
    setup_logging()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_warc_file, warc_files)

if __name__ == '__main__':
    main()