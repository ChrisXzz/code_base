# -*- coding: utf-8 -*-
import os
import cc3d
import nibabel as nib
import numpy as np
import math
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import argparse


def remove_small_components(seg_data, label, voxel_volume, min_radius):
    labeled, N = cc3d.connected_components(seg_data == label, connectivity=6, return_N=True)
    for i in range(1, N + 1):
        component = (labeled == i)
        if component.any():
            voxel = np.sum(component)
            volume = voxel * voxel_volume
            radius = (3 * volume / (4 * math.pi))**(1/3)
            #if radius <= min_radius and radius > 10:
            if radius <= min_radius:
                seg_data[component] = 0
    
    return seg_data

def keep_largest_component(seg_data, label):
    labeled, N = cc3d.connected_components(seg_data == label, connectivity=6, return_N=True)
    max_component = None
    max_size = 0
    for i in range(1, N + 1):
        component = (labeled == i)
        size = np.sum(component)
        if size > max_size:
            max_component = component
            max_size = size
    if max_component is not None:
        seg_data[seg_data == label] = 0
        seg_data[max_component] = label
    return seg_data

def process_segmentation(input_file, output_file):
    img = nib.load(input_file)
    data = img.get_fdata()
    voxel_volume = np.prod(img.header.get_zooms())
    data[data==24] = 1 #remove cyst, treat it as pancreas
    data[data==25] = 1 #remove PNET, treat it as pancreas
    data = keep_largest_component(data, 13)

    #for label in [23, 24, 25]:
    for label in [23]: #remove cyst mask
        data = remove_small_components(data, label, voxel_volume, min_radius=4)
        label_mask = (data == label)
        dilated_mask = binary_dilation(label_mask, iterations=1)
        if np.any(dilated_mask & (data == 13)):
            data[label_mask] = label
        else:
            data[label_mask] = 0

    data = data.astype(np.int8)
    processed_img = nib.Nifti1Image(data, img.affine)
    nib.save(processed_img, output_file)

def process_case(filename, input_dir, output_dir):
    input_file = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, filename)
    print(f"Processing {filename}...")
    process_segmentation(input_file, output_file)

def process_all_cases(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

    print('>> {} CPU cores are secured.'.format(int(cpu_count()*0.9)))
    with ProcessPoolExecutor(max_workers=int(cpu_count()*0.9)) as executor:
        futures = {
            executor.submit(process_case, filename, input_dir, output_dir): filename
            for filename in files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing cases'):
            filename = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process segmentation results with connected component analysis.")
    parser.add_argument('--input_dir', default='/ccvl/net/ccvl15/xinze_train/big_database/flagship/pano_health/', help='The directory containing input segmentation files.')
    parser.add_argument('--output_dir', default='/ccvl/net/ccvl15/xinze_train/big_database/flagship/pano_health_post/', help='The directory to save the processed segmentation files.')
    args = parser.parse_args()

    process_all_cases(args.input_dir, args.output_dir)
