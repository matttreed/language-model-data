import gzip
import random
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
from fastwarc import ArchiveIterator
import concurrent.futures
import os
from cs336_data.processing import detect_quality, detect_toxic, detect_nsfw, extract_text_from_html_bytes, mask_emails, mask_ip_addresses, mask_phone_numbers, identify_language, is_custom_gopher_quality, custom_filter
import logging
import logging.handlers
import multiprocessing
from queue import Queue
import threading
from datetime import datetime
import signal
import submitit
import argparse

WARC_WET_FILE_PATH = "/home/shared/CC-examples/example.warc.wet.gz"
WARC_FILE_PATH = "/home/shared/CC-examples/example.warc.gz"
WARC_DIR = '/home/shared/CC-MAIN-2023-50-warc-filtered'
DATA_DIR = "/home/c-mattreed/language-model-data/filtered_data"

def signal_handler(signal, frame):
    print("Signal received, stopping...")
    # You might want to set a global flag here that your threads check to exit cleanly
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
stop_requested = False

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
    return sorted(warc_files)

def log(message):
    pid = multiprocessing.current_process().pid
    tid = threading.get_ident()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp} - PID {pid} - TID {tid}: {message}")


def process_warc_file(filename, output_queue):
    if stop_requested:
        return

    base_name = os.path.basename(filename)

    log(f"Reading from {base_name}")

    total_count = 0
    lang_count = 0
    toxic_count = 0
    quality_count = 0
    gopher_count = 0
    pii_count = 0
    filtered_count = 0

    with gzip.open(filename, 'rb') as stream:
        archive_iterator = ArchiveIterator(stream)
        
        for record in archive_iterator:
            if stop_requested:
                break
            
            if record.headers.get('WARC-Type') == 'response':
                total_count += 1
                content = record.reader.read()
                text = extract_text_from_html_bytes(content)

                if detect_quality(text)[0] != "wiki":
                    quality_count += 1
                    continue

                if identify_language(text)[0] != "en":
                    lang_count += 1
                    continue

                if detect_toxic(text)[0] != "non-toxic":
                    toxic_count += 1
                    continue

                if not is_custom_gopher_quality(text):
                    gopher_count += 1
                    continue

                text = custom_filter(text)

                text, emails = mask_emails(text)
                text, ips = mask_ip_addresses(text)
                text, phones = mask_phone_numbers(text)

                if emails + ips + phones > 50:
                    pii_count += 1
                    continue

                filtered_count += 1

                output_text = text + "\n<|endoftext|>\n"

                output_queue.put(output_text)
    
    log(f"Total: {total_count}, Quality: {quality_count}, Lang: {lang_count}, Toxic: {toxic_count}, Custom Gopher: {gopher_count}, PII: {pii_count}, Passed Through: {filtered_count}")

def file_writer(queue, file_path):
    with open(file_path, 'a') as f:
        while True:
            message = queue.get()
            if message == "STOP":
                break
            f.write(message)
            f.flush()


# def main():
#     global stop_requested
#     warc_files = get_warc_files(WARC_DIR)
#     setup_logging()

    # output_queue = multiprocessing.Queue()
    # output_file_path = '/home/c-mattreed/language-model-data/filtered_data/raw.txt'
    
#     writer_process = multiprocessing.Process(target=file_writer, args=(output_queue, output_file_path))
#     writer_process.start()
    
#     try:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             futures = [executor.submit(process_warc_file, filename, output_queue) for filename in warc_files]
#             concurrent.futures.wait(futures)  # Wait for all futures to complete
#     except KeyboardInterrupt:
#         print("Keyboard interrupt received, notifying threads and processes...")
#     finally:
#         # Notify the writer process to stop
#         output_queue.put("STOP")
#         # Join the writer process to make sure it's finished
#         writer_process.join()
#         print("Cleanup complete.")

def process_files(file_chunk_i):
    file_chunk, i = file_chunk_i
    log(i)
    for filename in file_chunk:
        process_warc_file(filename)

def main():
    try:
        setup_logging()

        warc_files = get_warc_files(WARC_DIR)
        parser = argparse.ArgumentParser(description="Process some files.")

        parser.add_argument("start_index", type=int)
        parser.add_argument("num_docs", type=int)

        args = parser.parse_args()

        start = args.start_index
        num_docs = args.num_docs

        warc_files = warc_files[start:start + num_docs]

        log(f"PROCESSING {num_docs} DOCS STARTING AT INDEX {start}")

        manager = multiprocessing.Manager()
        output_queue = manager.Queue()
        output_file_path = '/home/c-mattreed/language-model-data/filtered_data/raw.txt'
        writer_process = multiprocessing.Process(target=file_writer, args=(output_queue, output_file_path))
        writer_process.start()

        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 10

        # Set start method for better cleanup in Unix when using Ctrl+C
        # multiprocessing.set_start_method('fork')

        with multiprocessing.Pool(num_cpus) as pool:
            # pool.map(lambda file: process_warc_file(file, output_queue), warc_files)
            pool.starmap(process_warc_file, [(file, output_queue) for file in warc_files])

        # Notify the writer process to stop
        output_queue.put("STOP")
    
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt (Ctrl+C), terminating workers...")
        if 'pool' in locals():
            pool.terminate()  # Terminate all worker processes immediately
            pool.join()       # Wait for the worker processes to terminate

    finally:
        # Always executed, regardless of error or not
        if writer_process.is_alive():
            output_queue.put("STOP")
            writer_process.join()  # Ensure the writer process also terminates properly
        print("Cleanup complete.")

# def main():
#     warc_files = get_warc_files(WARC_DIR)
#     setup_logging()

#     # Set up the multiprocessing queue and writer process
#     # output_queue = multiprocessing.Queue()
#     output_file_path = '/home/c-mattreed/language-model-data/filtered_data/raw.txt'

#     num_workers = 8  # Set based on your cluster setup or job submission parameters
#     chunks = [(warc_files[i::num_workers], i) for i in range(num_workers)]

#     # Executor setup for Slurm
#     executor = submitit.AutoExecutor(folder="submitit_logs")
#     executor.update_parameters(
#         slurm_partition='batch',
#         slurm_array_parallelism=8,  # This can be nodes or CPUs per node
#         timeout_min=60,  # Adjust as needed
#         mem_gb=10,
#         cpus_per_task=1,
#         nodes=2  # Request more nodes if needed
#     )

#     # Submit parallel jobs
#     job = executor.submit(process_files, chunks)

#     # Wait for the job to complete
#     job.result()  # Waits for the job to complete
#     print("Cleanup complete.")

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    main()