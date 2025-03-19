import os
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# label mapping
'''
label_mapping = [
    ('aorta.nii.gz', 1),
    ('adrenal_gland_left.nii.gz', 2),
    ('adrenal_gland_right.nii.gz', 3),
    ('common_bile_duct.nii.gz', 4),
    ('celiac_aa.nii.gz', 5),
    ('colon.nii.gz', 6),
    ('duodenum.nii.gz', 7),
    ('gall_bladder.nii.gz', 8),
    ('postcava.nii.gz', 9),
    ('kidney_left.nii.gz', 10),
    ('kidney_right.nii.gz', 11),
    ('liver.nii.gz', 12),
    ('pancreas.nii.gz', [13, 14, 23, 24, 25]),  
    ('pancreatic_duct.nii.gz', 14),
    ('superior_mesenteric_artery.nii.gz', 15),
    ('intestine.nii.gz', 16),
    ('spleen.nii.gz', 17),
    ('stomach.nii.gz', 18),
    ('veins.nii.gz', 19),
    ('renal_vein_left.nii.gz', 20),
    ('renal_vein_right.nii.gz', 21),
    ('cbd_stent.nii.gz', 22),
    ('_pancreatic_pdac.nii.gz', 23),
    ('_pancreatic_cyst.nii.gz', 24),
    ('_pancreatic_pnet.nii.gz', 25)
]
'''

label_mapping = [ ('_pancreatic_pdac.nii.gz', 23),
    ('_pancreatic_cyst.nii.gz', 24),
    ('_pancreatic_pnet.nii.gz', 25)
]

def process_case(file_name, source_dir, output_dir):
    if not file_name.endswith(".nii.gz"):
        return None

    case_name = file_name.replace(".nii.gz", "")
    input_path = os.path.join(source_dir, file_name)

    try:
        img = nib.load(input_path)
        data = img.get_fdata()

        case_output_dir = os.path.join(output_dir, case_name, "segmentations")
        os.makedirs(case_output_dir, exist_ok=True)

        saved = False  

        for label_name, label_value in label_mapping:
            mask = np.zeros_like(data, dtype=np.uint8)

            if isinstance(label_value, list):
                for val in label_value:
                    mask[data == val] = 1
            else:
                mask[data == label_value] = 1
            
            mask_nifti = nib.Nifti1Image(mask, img.affine, img.header)
            output_path = os.path.join(case_output_dir, label_name)
            nib.save(mask_nifti, output_path)
            saved = True

        return f"finished: {case_name}" if saved else f" {case_name}"

    except Exception as e:
        return f"process {case_name} error: {e}"

def process_all_cases(source_dir, output_dir):
    cases = [f for f in os.listdir(source_dir) if f.endswith(".nii.gz")]

    print(f">> finding {len(cases)} cases，starting...")
    print('>> {} CPU cores are secured.'.format(int(cpu_count()*0.9)))
    results = []
    with ProcessPoolExecutor(max_workers=int(cpu_count()*0.9)) as executor:
        futures = {executor.submit(process_case, case, source_dir, output_dir): case for case in cases}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            case = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                results.append(f"case {case} error: {e}")

    for res in results:
        print(res)

    print("All finished!")

def main():
    parser = argparse.ArgumentParser(description="split combined label")
    parser.add_argument("--source_dir", default = '/mnt/ccvl15/xinze_train/bigpaper/flagship/nnUNet/nnUNet_training/nnUNet_predictions/Dataset1334_RD/test/', help="input path")
    parser.add_argument("--output_dir", default = '/ccvl/net/ccvl15/xinze_train/big_database/flagship_mask/JHH_OUT/', help="output path")
    args = parser.parse_args()

    process_all_cases(args.source_dir, args.output_dir)

if __name__ == "__main__":
    main()

