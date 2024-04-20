from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd


all_shifts = []
all_data = []
#MAX_ATOMS = -1

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
    directory, file_name = os.path.split(file_path)
    basename, extension = os.path.splitext(file_name)
    gpr_file = os.path.join("all_structures_ghemical", f'{basename}.gpr')
    charges = []
    atoms = []
    bonds = np.zeros((0, 0), dtype=int)
    a = False
    c = False
    b = False
    with open(gpr_file, 'r') as file:
        for line in file:
            #print(line)
            if line.startswith('!Atoms'):
                a = True
                c = False
                b = False
                # Extract atoms
                num_atoms = int(line.split()[1])
                #print(num_atoms)
                #global MAX_ATOMS
                #MAX_ATOMS = max(MAX_ATOMS, num_atoms)
                atoms = np.zeros((num_atoms, 2), dtype=int)
            elif line.startswith('!Charges'):
                #print("charges")
                a = False
                b = False
                c = True
                # Extract charges
                charges = np.zeros(num_atoms, dtype=float)
            elif line.startswith('!Bonds'):
                c = False
                b = True
                a = False
                # Extract bonds
                bonds = np.zeros((num_atoms, 78), dtype=int)
                #Set to max number of atoms
            elif line.startswith('!'):
                a = False
                c = False
                b = False
                # Skip other information sections
                continue
            # Process charges, atoms, and bonds
            else:
                if c:
                    #print(line)
                    # Extract charges
                    atom_index, charge = map(float, line.split())
                    charges[int(atom_index)] = charge
                elif a:
                    # Extract atoms
                    atom_index, atomic_number = map(int, line.split())
                    atoms[int(atom_index)] = [atom_index, atomic_number]
                elif b:
                    # Extract bonds
                    atom_index1, atom_index2, bond_type = line.split()
                    atom_index1, atom_index2 = int(atom_index1), int(atom_index2)
                    bonds[atom_index1, atom_index2] = 1
                    bonds[atom_index2, atom_index1] = 1  # Assuming bonds are symmetric
    return np.array(atomic_coordinates), np.array(atom_types), np.array(charges), np.array(atoms), bonds

# Function to label encode atom types
def label_encode(atom_types):
    label_encoder = LabelEncoder()
    encoded_atom_types = label_encoder.fit_transform(atom_types)
    return encoded_atom_types, label_encoder.classes_
def train_gaussian(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kernel = DotProduct() + WhiteKernel()
    gaussian = GaussianProcessRegressor(kernel=kernel, random_state=42)

    scores = []
    r2_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gaussian.fit(X_train, y_train)

        score = gaussian.score(X_test, y_test)
        r2 = r2_score(y_test, gaussian.predict(X_test))
        mae = mean_absolute_error(y_test, gaussian.predict(X_test))

        scores.append(score)
        r2_scores.append(r2)
        mae_scores.append(mae)

    return gaussian, np.mean(scores), np.mean(r2_scores), np.mean(mae_scores)

# Function to process all XYZ files
def process_xyz_files(directory):
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        file_path = os.path.join(directory, filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        coordinates, atom_types, charges, atoms, bonds= read_xyz_file(file_path)
        encoded_atom_types, _ = label_encode(atom_types)

        data = np.column_stack((coordinates, encoded_atom_types, charges, atoms))
        data = np.column_stack((data, bonds))
        all_data.append(data)
        shifts = extract_shifts(shift_path)
        all_shifts.append(np.array(shifts))
directory = "all_structures"
process_xyz_files(directory)
#print(MAX_ATOMS)
print(len(all_data))
print(len(all_shifts))
#all_data stores all the xyz files with label-encoded atoms in a list
#each element of all_shifts is a list corresponding to the F NMR shifts of the corresponding xyz file in all_data

X_train, X_test, y_train, y_test = train_test_split(all_data, all_shifts, test_size=0.2, random_state=42)


# Align X_train and y_train
Xy_train_aligned = []
for X, shifts in zip(X_train, y_train):
    #print("Shifts", len(shifts))
    
    num_atoms = X.shape[0]
    replicated_shifts = np.tile(shifts, num_atoms)[:num_atoms]

    #print("Repl shifts:", len(replicated_shifts))
    #print("Num atoms:", num_atoms)
    # Ensure the shapes match
    if len(replicated_shifts) != num_atoms:
        # Adjust replication or handle mismatch
        raise ValueError("Mismatch between number of atoms and chemical shift values")

    # Concatenate X and replicated_shifts
    combined_data = np.hstack((X, replicated_shifts.reshape(-1, 1)))
    Xy_train_aligned.append(combined_data)

# Align X_test and y_test
Xy_test_aligned = []
for X, shifts in zip(X_test, y_test):
    num_atoms = X.shape[0]
    replicated_shifts = np.tile(shifts, num_atoms)[:num_atoms]

    # Ensure the shapes match
    if len(replicated_shifts) != num_atoms:
        # Adjust replication or handle mismatch
        raise ValueError("TEST DATA: Mismatch between number of atoms and chemical shift values")

    # Concatenate X and replicated_shifts
    combined_data = np.hstack((X, replicated_shifts.reshape(-1, 1)))
    Xy_test_aligned.append(combined_data)

Xy_train_aligned = np.concatenate(Xy_train_aligned)
Xy_test_aligned = np.concatenate(Xy_test_aligned)

# Separate input features and target variable for training
X_train_aligned = Xy_train_aligned[:, :-1]
y_train_aligned = Xy_train_aligned[:, -1]

# Separate input features and target variable for testing
X_test_aligned = Xy_test_aligned[:, :-1]
y_test_aligned = Xy_test_aligned[:, -1]

# Combine X_train_aligned and X_test_aligned into X_aligned
X_aligned = np.concatenate((X_train_aligned, X_test_aligned), axis=0)

# Combine y_train_aligned and y_test_aligned into y_aligned
y_aligned = np.concatenate((y_train_aligned, y_test_aligned), axis=0)
kernel = DotProduct() + WhiteKernel()
gaussian_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
gaussian_model.fit(X_train_aligned, y_train_aligned)

# Evaluate on test set
y_pred = gaussian_model.predict(X_test_aligned)

# Evaluate model
mse = mean_squared_error(y_test_aligned, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test_aligned, y_pred)
r2 = r2_score(y_test_aligned, y_pred)

print("Mean Absolute Error:", mae)
print("R-squared:", r2)


gaussian, accuracy, r2, mae = train_gaussian(X_aligned, y_aligned)

y_pred_mlp = gaussian.predict(X)

print("\n5-fold cross validation Results:")
print("R-squared:", r2)
print("Mean Absolute Error:", mae)
