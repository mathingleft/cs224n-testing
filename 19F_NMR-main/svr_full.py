import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

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
    #print(len(numbers))
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

# Function to process all XYZ files
def process_xyz_files(directory, max_num_atoms):
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        file_path = os.path.join(directory, filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        coordinates, atom_types = read_xyz_file(file_path)
        encoded_atom_types, _ = label_encode(atom_types)

        # Pad or truncate coordinates and encoded_atom_types to max_num_atoms
        if len(coordinates) < max_num_atoms:
            coordinates = np.pad(coordinates, ((0, max_num_atoms - len(coordinates)), (0, 0)), mode='constant', constant_values=0)
            encoded_atom_types = np.pad(encoded_atom_types, (0, max_num_atoms - len(encoded_atom_types)), mode='constant', constant_values=0)
        elif len(coordinates) > max_num_atoms:
            coordinates = coordinates[:max_num_atoms, :]
            encoded_atom_types = encoded_atom_types[:max_num_atoms]
        shifts = extract_shifts(shift_path)
        if (len(shifts) == 20):
            continue
        data = np.column_stack((coordinates, encoded_atom_types))
        all_data.append(data)
        pad_shifts = np.pad(shifts, (0, max_num_atoms - len(shifts)), mode='constant')
        all_shifts.append(pad_shifts)

def train_svr(X, y):
    svr = SVR(kernel='rbf')
    svr.fit(X, y)
    return svr

def predict_chemical_shifts(svr_model, X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    predictions = svr_model.predict(X_scaled)
    return predictions

max_num_atoms = 100

directory = "all_structures"
process_xyz_files(directory, max_num_atoms)
print(len(all_data))
print(len(all_shifts))
# Convert all_shifts into a single NumPy array
all_shifts = np.vstack(all_shifts)

# Ensure consistency in the number of samples by adjusting the train-test split
test_size = 0.2
x_train, x_test, shift_train, shift_test = train_test_split(all_data, all_shifts, test_size=test_size, random_state=42)
print(len(x_train))
print(len(shift_train))

# Flatten shift_train to make it a 1D array
shift_train_flat = shift_train.flatten()

print(len(x_train))
print(len(shift_train))

# Convert input_data to a single NumPy array
input_data = np.vstack(x_train)

# Print shapes for verification
print("Input data type:", type(input_data))
print("Input data shape:", input_data.shape)
print("Shift train type:", type(shift_train_flat))
print("Shift train shape:", shift_train_flat.shape)


# Initialize and fit SVR model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(input_data, shift_train_flat)
predictions = svr_model.predict(np.vstack(x_test))
print(predictions)
mae = mean_absolute_error(shift_test.flatten(), predictions)
r2 = r2_score(shift_test.flatten(), predictions)

print("Mean Absolute Error:", mae)
print("R-squared:", r2)
