import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

all_shifts = []
all_data = []

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
    for line in lines[coordinates_index:]:
        if line.strip():
            parts = line.split()
            atom_type = parts[0]
            x, y, z = map(float, parts[1:])
            atomic_coordinates.append([x, y, z])
            atom_types.append(atom_type)
    return np.array(atomic_coordinates), np.array(atom_types)

def label_encode(atom_types):
    label_encoder = LabelEncoder()
    encoded_atom_types = label_encoder.fit_transform(atom_types)
    return encoded_atom_types, label_encoder.classes_

def process_xyz_files(directory):
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        file_path = os.path.join(directory, filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        coordinates, atom_types = read_xyz_file(file_path)
        encoded_atom_types, _ = label_encode(atom_types)
        data = np.column_stack((coordinates, encoded_atom_types))
        all_data.append(data)
        shifts = extract_shifts(shift_path)
        all_shifts.append(np.array(shifts))

directory = "all_structures"
process_xyz_files(directory)
print(len(all_data))
print(len(all_shifts))

X_train, X_test, y_train, y_test = train_test_split(all_data, all_shifts, test_size=0.2, random_state=42)

Xy_train_aligned = []
for X, shifts in zip(X_train, y_train):
    num_atoms = X.shape[0]
    replicated_shifts = np.tile(shifts, num_atoms)[:num_atoms]
    if len(replicated_shifts) != num_atoms:
        raise ValueError("Mismatch between number of atoms and chemical shift values")
    combined_data = np.hstack((X, replicated_shifts.reshape(-1, 1)))
    Xy_train_aligned.append(combined_data)

Xy_test_aligned = []
for X, shifts in zip(X_test, y_test):
    num_atoms = X.shape[0]
    replicated_shifts = np.tile(shifts, num_atoms)[:num_atoms]
    if len(replicated_shifts) != num_atoms:
        raise ValueError("TEST DATA: Mismatch between number of atoms and chemical shift values")
    combined_data = np.hstack((X, replicated_shifts.reshape(-1, 1)))
    Xy_test_aligned.append(combined_data)

Xy_train_aligned = np.concatenate(Xy_train_aligned)
Xy_test_aligned = np.concatenate(Xy_test_aligned)

X_train_aligned = Xy_train_aligned[:, :-1]
y_train_aligned = Xy_train_aligned[:, -1]

X_test_aligned = Xy_test_aligned[:, :-1]
y_test_aligned = Xy_test_aligned[:, -1]

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 4, 5]
}

gb_model = GradientBoostingRegressor()

gb_model.fit(X_train_aligned, y_train_aligned)

y_pred = gb_model.predict(X_test_aligned)

mse = mean_squared_error(y_test_aligned, y_pred)
print("Mean Squared Error:", mse)
'''
(base) MacBook-Air-19:19F_NMR-main sophia$ python3 gradientboostingregression.py
501
501
Mean Squared Error: 1129.612510881723
Mean Absolute Error: 21.20528003788134
R-squared: -0.08453686150060458
'''

mae = mean_absolute_error(y_test_aligned, y_pred)
r2 = r2_score(y_test_aligned, y_pred)

print("Mean Absolute Error:", mae)
print("R-squared:", r2)

grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_aligned, y_train_aligned)

print("Best Parameters:", grid_search.best_params_)

best_gb_model = grid_search.best_estimator_
best_gb_model.fit(X_train_aligned, y_train_aligned)

y_pred = best_gb_model.predict(X_test_aligned)

mse = mean_squared_error(y_test_aligned, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test_aligned, y_pred)
r2 = r2_score(y_test_aligned, y_pred)

print("Mean Absolute Error:", mae)
print("R-squared:", r2)


