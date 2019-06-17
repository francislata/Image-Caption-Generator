"""This contains helper functions that is used by datasets."""

import os
from zipfile import ZipFile
from tqdm import tqdm
import requests

def download_from_url(url, dst):
    """
    Downloads the file from the given URL and add it to the destination

    Implementation from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py

    @param: url to download file
    @param: dst place to put the file
    """
    print('Downloading {}...'.format(url))

    file_size = int(requests.head(url).headers['Content-Length'])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        print('Done!\n')
        return file_size
    header = {'Range': 'bytes=%s-%s' % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f: #pylint: disable=invalid-name
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

    print('Done!\n')
    return file_size

def extract_zip(zip_file_path, destination_file_path):
    """Extracts the ZIP file to the given destination file path."""
    print('Extracting to {}...'.format(destination_file_path))

    with ZipFile(zip_file_path) as zip_file:
        name_list = zip_file.namelist()

        for file in tqdm(iterable=name_list, total=len(name_list)):
            zip_file.extract(file, path=destination_file_path)

    print('Done!\n')
