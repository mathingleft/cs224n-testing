import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

all_shifts = []
all_data = []

def extract_shifts(file_path):
    # Open the file and read its contents
    with open(file_path, "r") as file:
        file_content = file.read()

    # Find the index where "CHEMICAL SHIELDING SUMMARY" appears
    chemical_shielding_index = file_content.find("CHEMICAL SHIELDING SUMMARY")

    # Extract the text after "CHEMICAL SHIELDING SUMMARY"
    text_after_chemical_shielding = file_content[chemical_shielding_index:]

    # Split the text into lines and remove leading and trailing whitespace
    lines = text_after_chemical_shielding.split("\n")
    lines = [line.strip() for line in lines]
    lines = lines[1:]
    
    # Initialize a list to store the extracted numbers
    numbers = []

    # Loop through the lines and extract the numbers below "Isotropic"
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("Nucleus"):
            continue  # Skip header lines and separator lines
        if (line.strip().startswith("Maximum")):
            break
        if line.strip():  # Check if the line is not empty
            columns = line.split()  # Split the line into columns
            isotropic_value = columns[2]  # Extract the isotropic value (third column)
            numbers.append(float(isotropic_value))  # Convert to float and add to the list
    # Store the extracted numbers
    return numbers

# Function to read XYZ files and extract atomic coordinates
def read_xyz_file(file_path):
    coordinates_index = 2
    with open(file_path, "r") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if ("Coordinates from ORCA-job" in line):
                coordinates_index = i + 1
            i += 1
    # Extract atomic coordinates
    atomic_coordinates = []
    atom_types = []
    for line in lines[coordinates_index:]:
        if line.strip():  # Check if the line is not empty
            parts = line.split()
            atom_type = parts[0]
            x, y, z = map(float, parts[1:])
            atomic_coordinates.append([x, y, z])
            atom_types.append(atom_type)

    return np.array(atomic_coordinates), np.array(atom_types)

# Function to label encode atom types
def label_encode(atom_types):
    label_encoder = LabelEncoder()
    encoded_atom_types = label_encoder.fit_transform(atom_types)
    return encoded_atom_types, label_encoder.classes_

sums = 0
total = 0
# Function to process all XYZ files
def process_xyz_files(directory):
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        file_path = os.path.join(directory, filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        coordinates, atom_types = read_xyz_file(file_path)
        encoded_atom_types, _ = label_encode(atom_types)
        global sums
        global total
        data = np.column_stack((coordinates, encoded_atom_types))
        all_data.append(data)
        shifts = extract_shifts(shift_path)
        sums += sum(shifts)
        total += len(shifts)
        all_shifts.append(np.array(shifts))

directory = "all_structures"
process_xyz_files(directory)
print(len(all_data))
print(len(all_shifts))
print(sums/total)
#all_data stores all the xyz files with label-encoded atoms in a list
#each element of all_shifts is a list corresponding to the F NMR shifts of the corresponding xyz file in all_data