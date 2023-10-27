import os
import re
import requests
from uuid import uuid4
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from urllib.parse import urlparse, unquote
import hashlib
from concurrent.futures import ThreadPoolExecutor

def get_file_extension(url):
    """Get the file extension from a URL without query parameters."""
    url_without_query = url.split('?', 1)[0]
    return os.path.splitext(url_without_query)[-1]

def clean_filename(filename):
    """
    Sanitize a filename by removing unwanted characters.
    """
    s = re.sub(r'[^a-zA-Z0-9_\.-]', '_', filename)  # Replace unwanted chars with _
    return re.sub(r'(_)+', '_', s)  # Replace multiple consecutive _ with a single _

def get_filename_from_url(url):
    """
    Extract and clean the filename from a URL to include both path and basename.
    Then, hash the filename if it's too long.
    """
    parsed = urlparse(url)

    # Extract path parts and the basename
    path_parts = parsed.path.strip('/').split('/')
    sanitized_path = '_'.join([clean_filename(part) for part in path_parts])

    # If the filename is too long, hash it
    if len(sanitized_path) > 223:  # leaving some space for extension
        name, ext = os.path.splitext(sanitized_path)
        ext = ext[:4]  # limit extension to 4 chars
        hashed_name = hashlib.sha256(name.encode()).hexdigest()
        sanitized_path = f"{hashed_name}{ext}"

    return sanitized_path

def download_image(url, save_folder):
    """
    Downloads an image from the given URL and saves it to the provided folder.
    If the image file already exists, it doesn't download again.
    Returns the sanitized filename if successful, otherwise None.
    """
    sanitized_filename = get_filename_from_url(url)
    file_path = os.path.join(save_folder, sanitized_filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        return sanitized_filename

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Check if the response contains an image
        content_type = response.headers.get('content-type')
        if not content_type or 'image' not in content_type:
            return None

        # Save image
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return sanitized_filename

    except requests.RequestException:
        return None


def download_image_batch(urls, save_folder):
    """
    Download a batch of images.
    Returns a list of (url, filename) pairs for successfully downloaded images.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda u: (u, download_image(u, save_folder)), urls))
    return [(url, filename) for url, filename in results if filename]


def process_dataset(dataset, save_folder, split_name, batch_size=64):
    os.makedirs(save_folder, exist_ok=True)
    # new_data = []

    # Convert dataset to list for easier batch processing
    all_urls = list(dataset["URL"])

    # Use tqdm for progress bar
    for batch_start in tqdm(range(0, len(all_urls), batch_size), 
                            desc=f"Processing {split_name}", unit="batch"):
        
        batch_urls = all_urls[batch_start: batch_start+batch_size]

        # Parallelized download for this batch
        _ = download_image_batch(batch_urls, save_folder)

        # for url, sanitized_filename in downloaded_files:
        #    record = dataset.filter(lambda x: x["URL"] == url)[0]
        #    record["image"] = sanitized_filename
        #    new_data.append(record)

    # Convert back to Dataset format
    # return Dataset.from_dict({key: [item[key] for item in new_data] for key in new_data[0]})

def process_dataset_2(dataset, save_folder, split_name):
    new_data = []

    i = 0
    # Iterate directly over the dataset records
    for record in tqdm(dataset, desc=f"Processing {split_name}", unit="record"):
        # Get the sanitized filename corresponding to the URL
        sanitized_filename = get_filename_from_url(record["URL"])
        file_path = os.path.join(save_folder, sanitized_filename)

        if i > 293951: 
            break

        # check is the file exists in the local directory
        # if not skip it
        if not os.path.exists(file_path):
            i += 1
            continue
        
        # make sure the file is a file and not a directory
        if not os.path.isfile(file_path):
            i += 1
            continue

        # If the file exists in the local directory, update the dataset record

        record["URL"] = file_path  # Update URL field to store file path
        new_data.append(record)

        i += 1
          

    # Convert back to Dataset format
    return Dataset.from_dict({key: [item[key] for item in new_data] for key in new_data[0]})



def process_and_download(dataset_name):
    # Load your dataset
    datasets = load_dataset(dataset_name)

    # Process all splits
    new_datasets = {}
    image_folder = "downloaded_images"
    for split, dataset in datasets.items():
        print(f"Sample records from {split}:", dataset[:3])

        new_datasets[split] = process_dataset_2(dataset, image_folder, split)

    # Save the new datasets
    new_dataset_dict = DatasetDict(new_datasets)
    new_dataset_dict.save_to_disk("laion_6plus")

if __name__ == "__main__":
    dataset_name = "/mnt/newdrive/improved_aesthetics_6plus"
    process_and_download(dataset_name)



