import pandas as pd
import re


def ct_filename_permuted_search(ct_name_input, ct_name_data):
    # Split the input CT name into modality and date parts
    image_modality, date_str = ct_name_input.split('_')  # date_str = year-month-day
    year, month, day = map(int, date_str.split('-'))  # Extract year, month, day as integers
    
    # Prepare patterns dynamically with the actual values of day, month, and year
    patterns = [
        rf"im[-_. ]?ct[-_. ]?(?:0)?{day}[-_. ]?(?:0)?{month}[-_. ]?{year}",
        rf"im[-_. ]?ct[-_. ]?(?:0)?{year}[-_. ]?(?:0)?{month}[-_. ]?(?:0)?{day}",
        rf"ct[-_. ]?(?:0)?{day}[-_. ]?(?:0)?{month}[-_. ]?{year}",
        rf"ct[-_. ]?(?:0)?{year}[-_. ]?(?:0)?{month}[-_. ]?(?:0)?{day}"
    ]
    
    # Convert the CT name from the dataset to lowercase for case-insensitive comparison
    ct_name_data_lowercase = ct_name_data.lower()
    
    # Try each pattern to see if it matches the dataset CT name
    for pattern in patterns:
        if re.search(pattern, ct_name_data_lowercase):
            print(f"Image found in dataset")  # Show the matching pattern
            return True  # A match was found
    
    # If no match was found after trying all patterns
    print("Image not found in dataset")
    return False

""" # example usage:

ct_name = f"CT_19-07-04"
ct_data = f"IM CT_4.7.19"

ct_filename_permuted_search(ct_name, ct_data)

 """