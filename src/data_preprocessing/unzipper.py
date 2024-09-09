import os
import zipfile
import glob
from src.utils.config_loader import load_config

download_dir = '../' + load_config('../config/config.yaml')['download_dir']


def run():
    # First, extract all zip files
    for dirName, subdirList, fileList in os.walk(download_dir):
        print(f'Found directory: {dirName}')
        for fname in fileList:
            if fname.endswith('.zip'):
                print(f'\tFound file: {fname}')
                zip_path = os.path.join(dirName, fname)
                extracted_folder = os.path.join(dirName, os.path.splitext(fname)[0])

                if not os.path.exists(extracted_folder):
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(dirName)
                            print(f'\tUnzipped file: {fname}')
                    except zipfile.BadZipFile:
                        print(f'\tFile {fname} is not a valid zip file.')
                else:
                    print(f'\tFile {fname} already extracted.')

    # Then, delete zip files and non-CSV files
    for dirName, subdirList, fileList in os.walk(download_dir):
        for fname in fileList:
            file_path = os.path.join(dirName, fname)
            if fname.endswith('.zip'):
                os.remove(file_path)
                print(f'\tDeleted zip file: {fname}')
            elif not fname.endswith('.csv'):
                os.remove(file_path)
                print(f'\tDeleted non-CSV file: {fname}')

    print("Finished processing. Only CSV files remain.")