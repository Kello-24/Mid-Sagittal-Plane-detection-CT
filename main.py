import os
import pyesapi
import atexit
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.path import Path
import re
import csv

csv_filepath = r'P:/Desktop/For Yoel/ct_data_continued'

# IMPROVEMENTS:

# NO PRINT() OUTPUTS TO MAKE THE EXTRACTION PROCESS FASTER
# ONLY MAKE ONE (NOT THREE) NEW FOLDER IF PATIENT IS FOUND VIA CT FILENAME
# MATCH CT FILENAME WITH THE ONE ON CSV FILE USING POSSIBLE PERMUTATIONS
# EXTRACT IMPORTATNT STRUCTURES USING POSSIBLE NAMING CONVENTIONS
# GET A CSV FILE WITH PATIENT ID'S IF THE PATIENT FROM CSV IS NOT FOUND (CT FILE) OR IF NOT ALL STRUCTURES WERE FOUND (< 3)

# TODO:
# SAVE NIFTI VIA STRUCTURE NAMES USING FIXED NOMENCLATURE (GTVP, SPINAL CORD, MANDIBULA)


def ct_filename_permuted_search(ct_name_input, ct_name_data):
    
    # Split the input CT name into modality and date parts
    image_modality, date_str = ct_name_input.split('_')  # date_str = year-month-day
    year, month, day = map(int, date_str.split('-'))  # Extract year, month, day as integers
    
    # Prepare patterns dynamically with the actual values of day, month, and year
    patterns = [
        rf"(?:0)?{day}[-_. ]?(?:0)?{month}[-_. ]?{year}",
        rf"(?:0)?{year}[-_. ]?(?:0)?{month}[-_. ]?(?:0)?{day}",
        rf"(?:0)?{day}[-_. ]?(?:0)?{month}[-_. ]?{year}",
        rf"(?:0)?{year}[-_. ]?(?:0)?{month}[-_. ]?(?:0)?{day}"
    ]
    
    # Convert the CT name from the dataset to lowercase for case-insensitive comparison
    
    ct_name_data_lowercase = ct_name_data.lower()
    
    # Try each pattern to see if it matches the dataset CT name
    for pattern in patterns:
        if re.search(pattern, ct_name_data_lowercase):
            # print(f"Image found in dataset")  # Show the matching pattern
            return True  # A match was found
    
    # If no match was found after trying all patterns
    # print("Image not found in dataset")
    return False



def create_mask_from_contours(shape, contours):
    # Create a grid of points asdf
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    points = np.vstack((x.flatten(), y.flatten())).T

    # Convert each contour into a Path object
    paths = [Path(contour) for contour in contours]

    # Create a blank image
    img = np.zeros(shape, dtype=np.uint8)

    # Check if each point in the grid is inside any of the contours
    for path in paths:
        # Use contains_points for vectorized containment check
        grid = path.contains_points(points)
        grid = grid.reshape(shape)

        # Update the image: toggle the mask value for points inside this contourc
        img[grid] = 1 - img[grid]

    return img

print('Test Started')

save_path = 'D:/06_midline_extraction/'

app = pyesapi.CustomScriptExecutable.CreateApplication('TEST')
atexit.register(app.Dispose)

# Define structure name variants using regex patterns
structure_patterns = [
    r'body',  # Matches 'BODY'
    r'spinal[-_ ]?cord',  # Matches 'SpinalCord', 'spinal-cord', 'spinal cord', and 'spinal_cord'
    r'mandible',  # Matches 'mandible'
    r'mandibula',  # Matches 'mandibula'
    r'gtv[-_ ]?(p|pt|1)?'  # Matches 'gtvp', 'gtv pt', 'gtv_pt', 'gtv1', 'gtv-p', 'gtv-pt', etc.
]
# Define number of structures to find
num_structures = 3

# Compile the regex patterns for matching
structures = [re.compile(pattern, re.IGNORECASE) for pattern in structure_patterns]

# Initialize an empty list for missed patient data and found structures
missed_patients = [] # ct image was not found
structures_found = [] # less than 3 structures found
problem_patients = [] # ct image not found or less than 3 structures

df = pd.read_csv(os.path.join(csv_filepath, '.csv'), 
                 dtype={'P_SAP-NUMBER': str, 
                        'C_COURSEID': str, 
                        'PL_SETUP_ID': str, 
                        'IM_ID': str}, sep=';')

for index, row in df.iterrows():
    pat_id = str(row['P_SAP-NUMBER']).strip()
    
    try:
        patient = app.OpenPatientById(pat_id)
        
        print(f"Processing Patient ID: {pat_id}")

        found_structures = []  # List to keep track of found structures for this patient
        for pat_course in patient.Courses:
            looper = 0

            # Only generate folders if image.Id matches the value in the CSV row
            for pat_course_plan in patient.CoursesLot(pat_course.Id).PlanSetups:
                plan = patient.CoursesLot(pat_course.Id).PlanSetupsLot(pat_course_plan.Id)
                image = plan.StructureSet.Image

                # IDENTIFY DAY, MONTH AND YEAR OF CT USING DELIMITER
                # INCLUDE POSSIBLE PERMUTATIONS OF IMAGE ID: [DAY,MONTH,YEAR] OR [YEAR,MONTH,DAY]
                # CKECK ALL DIFFERENT WAYS THE CT IS NAMED: [CT_DAY.MONTH.YEAR, CT_DAY-MONTH-YEAR, IM CT_ ..., IM-CT_ ...]
                # -> SEE ct_filename_permuted_search()

                if ct_filename_permuted_search(row['IM_ID'], image.Id):
                    pat_folder_path = os.path.join(save_path, pat_id) # Only create one folder for the patient
                    course_path = os.path.join(pat_folder_path, pat_course_plan) # Only create one folder for the course

                    os.makedirs(pat_folder_path, exist_ok=True)
                    os.makedirs(course_path, exist_ok=True)

                    # Process the image (assumes img is defined)
                    sitk.WriteImage(img, os.path.join(course_path, 'image.nii.gz'))
                    print('Image Saved')

                    # Process structures
                    for structure in plan.StructureSet.Structures:
                        # Check if the structure.Id matches any regex pattern defined in structures
                        if any(regex.search(structure.Id) for regex in structures):
                            found_structures.append(structure.Id)  # Collect found structure ID
                            structure_set = np.zeros(array.shape)

                            for z_slice in range(array.shape[2]):
                                # print(f'Patient {pat_id}, Structure {structure.Id}, Progress: {round(z_slice / array.shape[2] * 100, 3)}%', end='\r')

                                rt_struct = plan.StructureSet.StructuresLot(structure.Id)
                                contour = rt_struct.GetContoursOnImagePlane(z_slice)
                                list_contour_coord_per_slice = []

                                for pt_list in contour:
                                    contour_per_slice = [[(pt.y - y_origin) / y_res, (pt.x - x_origin) / x_res] for pt in pt_list]
                                    list_contour_coord_per_slice.append(contour_per_slice)

                                structure_set[:, :, z_slice] = create_mask_from_contours(
                                    shape=(array.shape[0], array.shape[1]),
                                    contours=list_contour_coord_per_slice
                                )

                            # Save the structure mask
                            img = sitk.GetImageFromArray(structure_set.T)
                            img.SetOrigin(np.array(origin_vec))
                            img.SetSpacing(np.array(spacing))
                            img.SetDirection(dir_vec)
                            sitk.WriteImage(img, os.path.join(course_path, f'mask_{structure.Id}.nii.gz'))
                            # print(f"Mask saved for structure {structure.Id}")

                else:
                    # If filename check fails, add to missed patients list
                    missed_patients.append(pat_id)
                    problem_patients.append({'P_SAP-NUMBER': pat_id, 'C_COURSEID': '', 'PL_SETUP_ID': '', 'IM_ID': image.Id})

        # Add the patient ID and found structures to the list if fewer than 3 structures were found
        if len(found_structures) < num_structures:
            structures_found.append({'Patient_ID': pat_id, 'Found_Structures': found_structures})
            problem_patients.append({'P_SAP-NUMBER': pat_id, 'C_COURSEID': '', 'PL_SETUP_ID': '', 'IM_ID': image.Id})

    except (AttributeError, RuntimeError, NotADirectoryError) as e:
        print(f"An error occurred: {e}")

    finally:
        app.ClosePatient()


# WRITING NEW CSV FILES ----------------------------------------------------------------------------------------------------------------

# After processing all patients, save the missed patient IDs and found structures to a CSV
output_data = []
output_data.append({'Patient_ID': 'Missed', 'Found_Structures': ''})  # Add header for missed patients
for patient_id in missed_patients:
    output_data.append({'Patient_ID': patient_id, 'Found_Structures': ''})

# Append structures found information
for structure_info in structures_found:
    output_data.append({'Patient_ID': structure_info['Patient_ID'], 'Found_Structures': ', '.join(structure_info['Found_Structures'])})

# Save to CSV file
output_csv_path = os.path.join(save_path, 'missing_patient_structure_data.csv')
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Patient_ID', 'Found_Structures']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_data)

# WRITE A NEW CSV, IN THE SAME FORMAT AS THE INPUT CSV, WITH ONLY THE PATIENTS WHICH EITHER THE CT IMAGE 
# COULD NOT BE MATCHED OR NOT ALL THREE STRUCTURES WERE FOUND 

output_csv_path_problem_patients = os.path.join(save_path, 'problem_patients_patid_imid.csv')
with open(output_csv_path_problem_patients, 'w', newline='') as csvfile:
    fieldnames = ['P_SAP-NUMBER', 'C_COURSEID', 'PL_SETUP_ID', 'IM_ID']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(problem_patients)


# print(f"Data saved to {output_csv_path}")



# OLD VERSION

""" df = pd.read_csv('P:/Desktop/For Yoel/ct_data_continued.csv', 
                 dtype={'P_SAP-NUMBER': str, 
                        'C_COURSEID': str, 
                        'PL_SETUP_ID': str, 
                        'IM_ID': str}, sep = ';')


for index, row in df.iterrows():
        
    pat_id = str(row['P_SAP-NUMBER']).strip()
        
    pat_folder_path = save_path + pat_id.strip()
    if not os.path.exists(pat_folder_path):
        os.makedirs(pat_folder_path)
    	
    patient = app.OpenPatientById(pat_id)
    
    print(pat_id)
    try:
        for pat_course in patient.Courses:
            pat_course_folder_path = os.path.join(pat_folder_path, pat_course.Id)
            if not os.path.exists(pat_course_folder_path):
                os.makedirs(pat_course_folder_path)
            try:
                looper = 0
                for pat_course_plan in patient.CoursesLot(pat_course.Id).PlanSetups:
                    pat_course_plan_folder_path = os.path.join(pat_course_folder_path, pat_course_plan.Id)
                    if re.match("^[A-Za-z0-9_-]*$", pat_course_plan_folder_path):
                        pass
                    else:
                        looper += 1
                        pat_course_plan_folder_path = 'D:/06_midline_extraction/' + str(pat_id) +'/renamed' + str(looper)
                    if not os.path.exists(pat_course_plan_folder_path):
                        os.makedirs(pat_course_plan_folder_path)
        
                    plan = patient.CoursesLot(pat_course.Id).PlanSetupsLot(pat_course_plan.Id)
                    try:
                        image = plan.StructureSet.Image
                        if image.Id == row['IM_ID']:
            
                            origin = image.Origin
                            origin_vec = [origin.x, origin.y, origin.z]
                            dir_vec = [image.XDirection.x, image.XDirection.y, image.XDirection.z,
                                       image.YDirection.x, image.YDirection.y, image.YDirection.z,
                                       image.ZDirection.x, image.ZDirection.y, image.ZDirection.z]
                            spacing = [image.XRes, image.YRes, image.ZRes]
                
                            x_res = image.XRes
                            y_res = image.YRes
                            x_origin = origin.x
                            y_origin = origin.y
                
                            array = plan.StructureSet.Image.np_array_like()-1024
                
                            img = sitk.GetImageFromArray(array)
                            img.SetOrigin(np.array(origin_vec))
                            img.SetSpacing(np.array(spacing))
                            img.SetDirection(dir_vec)
                        
                        
                            try:
                                sitk.WriteImage(img, pat_course_plan_folder_path+'/image.nii.gz')
                                print('Image Saved')
                
                                structure_set = {}
                                for structure in plan.StructureSet.Structures:
                                    if structure.Id in structure_list or 'gtv' in structure.Id.lower():
                                        structure_set = np.zeros(array.shape)
                
                                        for z_slice in range(array.shape[2]):
                                            print(f'Current Patient {pat_id}; Current structure {structure.Id} Current progress {round(z_slice/array.shape[2]*100,3)} %                   ', end='\r')
                
                                            rt_struct = plan.StructureSet.StructuresLot(structure.Id)
                                            contour = rt_struct.GetContoursOnImagePlane(z_slice)
                                            list_contour_coord_per_slice = []
                        		
                                            for pt_list in contour:
                                                contour_per_slice = [[(pt.y - y_origin) / y_res, (pt.x - x_origin) / x_res] for pt in pt_list]
                                                list_contour_coord_per_slice.append(contour_per_slice)
                                            structure_set[:,:,z_slice] = create_mask_from_contours(shape=(array.shape[0], array.shape[1]), contours=list_contour_coord_per_slice)
                    
                                    
                                        img = sitk.GetImageFromArray(structure_set.T)
                                        img.SetOrigin(np.array(origin_vec))
                                        img.SetSpacing(np.array(spacing))
                                        img.SetDirection(dir_vec)
                                        sitk.WriteImage(img, pat_course_plan_folder_path + '/mask_' + str(structure.Id) + '.nii.gz')
                                
                            except (RuntimeError, NotADirectoryError) as e:
                                                print(f"An error occurred: {e}")
                                                #write to the log file
                    except (AttributeError) as e:
                        print(f"An error occurred: {e}")
            except (AttributeError) as e:
                print(f"An error occurred: {e}")
    except (AttributeError) as e:
        print(f"An error occurred: {e}")

    app.ClosePatient() """