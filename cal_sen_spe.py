# -*- coding: utf-8 -*-
import os
import nibabel as nib
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_case(file_path):
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()

        case_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]
        if np.any(np.isin(data, [23, 25])):
            return case_name, True  
        else:
            return case_name, False  
    except Exception as e:
        return f"Error processing {file_path}: {e}", None

def process_cases(input_dir, output_with_labels, output_without_labels):
    files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.endswith(".nii.gz")]

    print(f">> Found {len(files)} cases. Starting processing...")

    cases_with_labels = []
    cases_without_labels = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_case, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
            file_path = futures[future] 
            try:
                case, has_labels = future.result()
                if has_labels is True:
                    cases_with_labels.append(case)
                elif has_labels is False:
                    cases_without_labels.append(case)
                else:
                    print(f"Error in {file_path}: {case}")  
            except Exception as e:
                print(f"Exception in {file_path}: {e}")

    with open(output_with_labels, "w") as f:
        for case in cases_with_labels:
            f.write(f"{case}\n")

    with open(output_without_labels, "w") as f:
        for case in cases_without_labels:
            f.write(f"{case}\n")

    print(f"Total {len(cases_with_labels)} cases with labels 23, 24, 25 saved to {output_with_labels}")
    print(f"Total {len(cases_without_labels)} cases without labels 23, 24, 25 saved to {output_without_labels}")

def main():

    parser = argparse.ArgumentParser(description="检查 NIfTI 文件是否包含 label 23, 24, 25，并分类保存")
    parser.add_argument("--input_dir", default = '/ccvl/net/ccvl15/xinze_train/big_database/flagship/pano_health_post/', help="NIfTI 文件存放路径")
    parser.add_argument("--output_with_labels", default = '/ccvl/net/ccvl15/xinze_train/big_database/flagship/pano_health_with_lables.txt', help="包含标签的案例输出路径")
    parser.add_argument("--output_without_labels", default = '/ccvl/net/ccvl15/xinze_train/big_database/flagship/pano_health_without_labels.txt', help="不包含标签的案例输出路径")
    args = parser.parse_args()

    process_cases(args.input_dir, args.output_with_labels, args.output_without_labels)

if __name__ == "__main__":
    main()

