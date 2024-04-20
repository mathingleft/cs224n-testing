import math
import os 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score

##DATA PROCESSING BEGINS
def extract_shifts(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    chemical_shielding_index = file_content.find("CHEMICAL SHIELDING SUMMARY")
    text_after_chemical_shielding = file_content[chemical_shielding_index:]

    lines = text_after_chemical_shielding.split("\n")
    lines = [line.strip() for line in lines]
    lines = lines[1:]
    numbers = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("Nucleus"):
            continue 
        if (line.strip().startswith("Maximum")):
            break
        if line.strip(): 
            columns = line.split() 
            isotropic_value = columns[2]
            numbers.append(float(isotropic_value)) 
    return numbers

def read_ghemical(file_path):
    _, file_name = os.path.split(file_path)
    basename, _ = os.path.splitext(file_name)
    gpr_file = os.path.join("all_structures_ghemical", f'{basename}.gpr')
    charges = []
    atoms = []
    a = False
    c = False
    with open(gpr_file, 'r') as file:
        for line in file:
            if line.startswith('!Atoms'):
                a = True
                c = False
            elif line.startswith('!Charges'):
                a = False
                c = True
            elif line.startswith('!'):
                a = False
                c = False
                continue
            else:
                if c:
                    _, charge = map(float, line.split())
                    charges.append(charge)
                elif a:
                    _, atomic_number = map(int, line.split())
                    atoms.append(atomic_number)
    return charges, atoms

def read_xyz_file(file_path):
    coordinates_index = 2
    with open(file_path, "r") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if ("Coordinates from ORCA-job" in line):
                coordinates_index = i + 1
            i += 1
    atomic_coordinates = []
    atom_types = []
    charges, atomic_numbers = read_ghemical(file_path)
    for line in lines[coordinates_index:]:
        if line.strip(): 
            parts = line.split()
            atom_type = parts[0]
            x, y, z = map(float, parts[1:])
            atomic_coordinates.append([x, y, z])
            atom_types.append(atom_type)
    f_neighbors_3 = []
    f_neighbors_2 = []
    index = 0
    for i in range(len(atom_types)):
        if (atom_types[i] is "F"):
            x = atomic_coordinates[i][0]
            y = atomic_coordinates[i][1]
            z = atomic_coordinates[i][2]
            f_neighbors_3.append(list())
            f_neighbors_2.append(list())
            for j in range(len(atom_types)):
                if (j != i):
                    dist_x = (atomic_coordinates[j][0] - x) ** 2
                    dist_y = (atomic_coordinates[j][1] - y) ** 2
                    dist_z = (atomic_coordinates[j][2] - z) ** 2
                    distance = math.sqrt(dist_x + dist_y + dist_z)
                    if (distance <= 3):
                        f_neighbors_3[index].append([atom_types[j], distance, charges[j], atomic_numbers[j]])
                    if (distance <= 2):
                        f_neighbors_2[index].append([atom_types[j], distance, charges[j], atomic_numbers[j]])
            index += 1
    return f_neighbors_3, f_neighbors_2

def count_max_F(directory):
    maximum = -1
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        shifts = extract_shifts(shift_path)
        maximum = max(maximum, len(shifts))
    return maximum

def process_xyz_files(directory, maximum):
    data = pd.DataFrame(columns=['neighbors', 'chemical_shift'])
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        file_path = os.path.join(directory, filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        fn3, fn2 = read_xyz_file(file_path)
        ##I USED FN3 ONLY TO TEST THE DATA PROCESSING, BUT YOU SHOULD RUN
        ##THE MODELS USING FN3 ONLY AND FN2 ONLY TO COMPARE THEIR RESULTS  
        shifts = extract_shifts(shift_path)
        for i in range(len(fn3)):
            row = {
                'neighbors': fn3[i], ##CHANGE FN3 TO FN2 TO TEST THE OTHER RESULT (Note: SVR got worse results with FN2)
                'chemical_shift': shifts[i]  
            }
            data = data._append(row, ignore_index=True)
    return data
directory = "all_structures"
maximum = count_max_F(directory)
data = process_xyz_files(directory, maximum)
print(data.head())
print(data.shape)

def flatten_neighbors_with_types(neighbors):
    flattened_distances = []
    flattened_neighbors = []
    flattened_charges = []
    flattened_nums = []
    max_length = max(len(neighbor_list) for neighbor_list in neighbors)
    for neighbor_list in neighbors:
        distances = []
        neighbor_types = []
        charges = []
        nums = []
        for neighbor, distance, charge, num in neighbor_list:
            distances.append(distance) 
            neighbor_types.append(neighbor) 
            charges.append(charge)
            nums.append(num)
        padding_length = max_length - len(distances)
        distances.extend([0] * padding_length)
        neighbor_types.extend([0] * padding_length)
        charges.extend([0] * padding_length)
        nums.extend([0] * padding_length)
        flattened_distances.append(distances)
        flattened_neighbors.append(neighbor_types)
        flattened_charges.append(charges)
        flattened_nums.append(nums)
    return flattened_distances, flattened_neighbors, flattened_charges, flattened_nums

distances, neighbors, charges, nums = flatten_neighbors_with_types(data['neighbors'])

encoder = OneHotEncoder(sparse=False)
atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types

distances_encoded = np.array(distances)
charges = np.array(charges)
nums = np.array(nums)

X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums])

y = data['chemical_shift']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
##DATA PROCESSING COMPLETE 

##FOR REGRESSION MODELS, ALL YOU NEED TO CHANGE IS THIS PART (FROM SVR TO SOME OTHER MODEL). 
##FEEL FREE TO IMPLEMENT K-FOLD CROSS VALIDATION AND FINE-TUNING AS NEEDED
# Train SVR model
svr = SVR(kernel='rbf', C=100, gamma='auto')  # Adjust parameters as needed
svr.fit(X_train, y_train)

# Predict on testing data
y_pred = svr.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R-squared:", r2)