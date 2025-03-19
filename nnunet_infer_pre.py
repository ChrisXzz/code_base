# -*- coding: utf-8 -*-
import os
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

def is_valid_case(case_folder):
    if case_folder.startswith("BDMAP_"):
        try:
            #case_number = int(case_folder.split("_")[-1])
            case_number = int(case_folder.split("_")[-1])
            return 12001 <= case_number <= 16000
        except ValueError:
            return False
    return False

def copy_ct_file(case_folder, source_dir, destination_dir):
    case_path = os.path.join(source_dir, case_folder)
    ct_file_path = os.path.join(case_path, 'ct.nii.gz')
    if os.path.exists(ct_file_path):
        new_file_name = f"{case_folder}_0000.nii.gz"
        destination_file_path = os.path.join(destination_dir, new_file_name)
        shutil.copy2(ct_file_path, destination_file_path)
        return f"Copied {ct_file_path} to {destination_file_path}"
    else:
        return f"No ct.nii.gz found in {case_path}"

def organize_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    tasks = [case_folder for case_folder in os.listdir(source_dir) 
             if os.path.isdir(os.path.join(source_dir, case_folder)) and is_valid_case(case_folder)]
    
    results = []

    print('>> {} CPU cores are secured.'.format(cpu_count()))
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {
            executor.submit(copy_ct_file, task, source_dir, destination_dir): task
            for task in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing cases'):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing folder {task}: {e}")

    for result in results:
        print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copy and rename ct.nii.gz files from source to destination.")
    parser.add_argument('--source_dir', default='/mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro/',
                        help='The source directory containing the case folders.')
    parser.add_argument('--destination_dir', default='/ccvl/net/ccvl15/xinze_train/bigpaper/flagship/nnUNet/nnUNet_training/nnUNet_eval/Dataset1328_rpro4/imagesTs/',
                        help='The destination directory to save the copied files.')
    args = parser.parse_args()

    organize_files(args.source_dir, args.destination_dir)


