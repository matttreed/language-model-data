import gzip
import random
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
from fastwarc import ArchiveIterator

WARC_WET_FILE_PATH = "/home/shared/CC-examples/example.warc.wet.gz"
WARC_FILE_PATH = "/home/shared/CC-examples/example.warc.gz"

# def sample_warc_records(file_path = WARC_FILE_PATH, sample_size = 1):
#     with gzip.open(file_path, 'rb') as stream:
#         archive_iterator = ArchiveIterator(stream)
        
#         all_records = []
#         for i, record in enumerate(archive_iterator):
#             if i > 1000:
#                 break
#             print(record.rec_type)
#             if record.rec_type == 'response':
#                 all_records.append(record)
        
#         sampled_records = random.sample(all_records, sample_size)
        
#         sampled_content = []
#         for record in sampled_records:
#             content = record.content_stream().read()
#             sampled_content.append(content)
        
#         return sampled_content


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