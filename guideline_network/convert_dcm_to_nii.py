import os
import SimpleITK as sitk
import trimesh
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import re
import argparse


#############################################################################################

def find_mri_sequences(patient_dir):
    """
    Identifies and returns the paths to specific MRI sequences (T2-weighted, ADC, and DWI) 
    within a given patient directory.
    The function traverses the directory structure, looking for subdirectories containing 
    DICOM files and matching specific naming patterns for the desired MRI sequences.
    Args:
        patient_dir (str): Path to the root directory of the patient's MRI data.
    Returns:
        tuple: A tuple containing three elements:
            - t2w_path (str or None): Path to the T2-weighted (T2W) sequence directory, 
              or None if not found.
            - adc_path (str or None): Path to the Apparent Diffusion Coefficient (ADC) 
              sequence directory, or None if not found.
            - dwi_path (str or None): Path to the Diffusion-Weighted Imaging (DWI) 
              sequence directory, or None if not found.
    Notes:
        - The function skips directories named "3D Rendering".
        - A directory is considered valid if it contains files with the ".dcm" extension.
        - T2W is identified by the presence of "t2" in the directory name.
        - ADC is identified by the presence of both "ep2d" and "adc" in the directory name.
        - DWI is identified by the presence of "ep2d" but not "adc" in the directory name.
    """
    
    t2w_path = None
    adc_path = None
    dwi_path = None

    for root, dirs, files in os.walk(patient_dir):
        # Skip non-DICOM directories
        if "3D Rendering" in root:
            continue

        if not files or not any(file.endswith(".dcm") for file in files):
            continue

        dirname = os.path.basename(root).lower()

        # Match T2W
        if "t2" in dirname and t2w_path is None:
            t2w_path = root

        # Match ADC: must have both "ep2d" and "adc"
        elif "ep2d" in dirname and "adc" in dirname and adc_path is None:
            adc_path = root

        # Match DWI: must have "ep2d" but not "adc" to avoid overlapping with ADC
        elif "ep2d" in dirname and "adc" not in dirname and dwi_path is None:
            dwi_path = root

    return t2w_path, adc_path, dwi_path



#############################################################################################


def load_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_IDs:
        raise Exception(f"No DICOM series found in {dicom_dir}")

    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    return image


#############################################################################################

def save_file(image, output_path):
    # Convert SimpleITK image to NumPy array
    arr = sitk.GetArrayFromImage(image)
    
    # Ensure output_path ends with .npy
    if not output_path.endswith(".npy"):
        output_path += ".npy"

    # Save the array
    np.save(output_path, arr)
#############################################################################################


def process_modality(modality, stl_file, dicom_dir, base_path):
    # Extract the filename from the path
    filename = os.path.basename(stl_file)
    # Regular expression to extract the desired part
    match = re.match(r"^([^-]+(?:-[^-]+){0,5})-seriesUID", filename)
    extracted_part = match.group(1)
    # Insert the modality name between the patient info and Target
    new_filename = re.sub(r"(Prostate-MRI-US-Biopsy-\d+)-Target(\d+)", r"\1_" + modality + r"_Target\2", extracted_part)
        
    image = load_dicom_series(dicom_dir)
    save_file(image, f"{base_path}/{new_filename}.npy")

#############################################################################################


def read_stl_record_csv(file_path):
    """
    Reads a CSV file containing STL records and returns a list of dictionaries.
    Each dictionary contains the information of one STL record.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        list: A list of dictionaries, each representing an STL record.
    """

    df = pd.read_csv(file_path)
    stl_paths = df["STL_dir"].tolist()
    patient_uids = df["patient_ID"].tolist()
    
    return stl_paths, patient_uids
#############################################################################################


def process_dataset(stl_record_file_path, manifest_dir, stl_dir, output_dir):
    
    stl_files, patient_uids = read_stl_record_csv(stl_record_file_path)
    for stl_file, patient_uid in zip(stl_files, patient_uids):
        stl_file_path = os.path.join(stl_dir, stl_file)
        patient_dir = os.path.join(manifest_dir, patient_uid)
        
        t2w, adc, dwi = find_mri_sequences(patient_dir)
        
        patient_output_dir = os.path.join(output_dir, patient_uid)
        os.makedirs(patient_output_dir, exist_ok=True)
        print("Processing patient:", patient_uid)
        if t2w:
            process_modality("T2W", stl_file_path, t2w, patient_output_dir)
        if adc:
            process_modality("ADC", stl_file_path, adc, patient_output_dir)
        if dwi:
            process_modality("DWI", stl_file_path, dwi, patient_output_dir)
        
#############################################################################################


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process dataset for Prostate Cancer Research - Converting DICOM to NIfTI")
    
    # Add arguments for the directories
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help="The base directory where the dataset is stored")
    parser.add_argument('--stl_record_file', type=str, required=True, 
                        help="Path to the STL record CSV file")
    parser.add_argument('--output_dir', type=str, default="prostate/case_input_2", 
                        help="Directory to save the output (default: 'prostate/case_input_2')")

    # Parse arguments
    args = parser.parse_args()

    # Construct the full paths for the required directories
    manifest_dir = os.path.join(args.dataset_dir, "manifest-1694710246744/Prostate-MRI-US-Biopsy")
    stl_dir = os.path.join(args.dataset_dir, "PKG - Prostate-MRI-US-Biopsy-STL/Prostate-MRI-US-Biopsy/STLs")
    
    # Call the function with the arguments
    process_dataset(args.stl_record_file, manifest_dir, stl_dir, args.output_dir)

if __name__ == "__main__":
    main()