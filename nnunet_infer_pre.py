# -*- coding: utf-8 -*-
import os
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

def copy_ct_file(case_folder, source_dir, destination_dir):
    case_path = os.path.join(source_dir, case_folder)
    ct_file_path = os.path.join(case_path, 'ct.nii.gz')
    #ct_file_path = os.path.join(case_path, 'combined_labels.nii.gz')
    if os.path.exists(ct_file_path):
        new_file_name = f"{case_folder}_0000.nii.gz"
        destination_file_path = os.path.join(destination_dir, new_file_name)
        shutil.copy2(ct_file_path, destination_file_path)
        return f"Copied {ct_file_path} to {destination_file_path}"
    else:
        return f"No ct.nii.gz found in {case_path}"

def organize_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    tasks = [case_folder for case_folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, case_folder))]
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
    parser.add_argument('--source_dir', default= '/media/chrisxzz/PortableSSD/bigpaper/felix_rd', help='The source directory containing the case folders.')
    parser.add_argument('--destination_dir', default= '/home/chrisxzz/bigpaper/nnUNet/nnUNet_training/nnUNet_eval/Dataset1334_RD/imagesTs/', help='The destination directory to save the copied files.')
    args = parser.parse_args()

    organize_files(args.source_dir, args.destination_dir)

