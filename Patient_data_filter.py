import pandas as pd
import re
import os

def patient_ID_extension(csv_source_filepath, csv_new_folderpath, id_length_check=None):
    # Read the CSV file
    df = pd.read_csv(csv_source_filepath, sep=';')

    # Filter patient IDs with fewer digits than specified by id_length_check
    if id_length_check is not None:
        filtered_df = df[df['P_SAP-NUMBER'].apply(lambda x: len(str(x)) < id_length_check and str(x).isdigit())]
        
        # Save filtered patient IDs to a new CSV
        filtered_df.to_csv(os.path.join(csv_new_folderpath, f"patient_id_smth_{id_length_check}_dgts.csv"), index=False, sep=';')

    # Fill in missing zeros in front of ID if length is less than 8
    df['P_SAP-NUMBER'] = df['P_SAP-NUMBER'].apply(lambda x: str(x).zfill(8) if len(str(x)) < 8 else str(x))

    # Save adjusted patient IDs to a new CSV
    df.to_csv(os.path.join(csv_new_folderpath, "adjusted_patient_ids.csv"), index=False, sep=';')


""" # Example usage
csv_source_path = r'/home/loriskeller/Documents/Master Project/VS/Data_extract_and_midline/ct_data.csv'
csv_new_folderpath = r'/home/loriskeller/Documents/Master Project/VS/Data_extract_and_midline'

patient_ID_extension(csv_source_path, csv_new_folderpath, id_length_check=8)
 """