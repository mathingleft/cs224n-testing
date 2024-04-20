import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

import pandas as pd

def process_xyz_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    atom_labels = []
    coordinates = []

    for line in lines[2:]:
        data = line.split()
        atom_labels.append(data[0])
        coordinates.append([float(coord) for coord in data[1:4]])

    return atom_labels, coordinates

def process_out_file(file_path):
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

def label_encode_atoms(atom_labels):
    label_encoder = LabelEncoder()
    encoded_atoms = label_encoder.fit_transform(atom_labels)
    return encoded_atoms

def load_processed_data(xyz_directory, shifts_directory):
    atom_labels = []
    coordinates = []
    shifts = []

    for filename in os.listdir(xyz_directory):
        if filename.endswith('.xyz'):
            xyz_file = os.path.join(xyz_directory, filename)
            shifts_file = os.path.join(shifts_directory, filename.replace('.xyz', '.out'))

            atom_labels_temp, coordinates_temp = process_xyz_file(xyz_file)
            shifts_temp = process_out_file(shifts_file)
            #print("Former shifts_length: ", len(shifts_temp))
            #print("Before padding, shifts=", shifts_temp)
            max_length = max(len(atom_labels_temp), len(shifts_temp))
            atom_labels_temp = atom_labels_temp[:max_length]
            coordinates_temp = coordinates_temp[:max_length]
            #shifts_temp = shifts_temp[:min_length]
            #mean_val = sum(shifts_temp)/len(shifts_temp)
        
            for i in range(max_length - len(shifts_temp)):
                #shifts_temp.append(mean_val)
                shifts_temp.append(278.8152962962958)
            #print("Shifts_length: ", len(shifts_temp))
            #print("AFTER padding, shifts=", shifts_temp)
            #print("Atom_labels: ", len(atom_labels_temp))
            
            atom_labels.extend(atom_labels_temp)
            coordinates.extend(coordinates_temp)
            shifts.extend(shifts_temp)

    encoded_atoms = label_encode_atoms(atom_labels)
    coordinates = np.array(coordinates)
    data = {
        'Atom Labels': encoded_atoms,
        'X': coordinates[:, 0],
        'Y': coordinates[:, 1],
        'Z': coordinates[:, 2],
        'Shifts': shifts
    }
    df = pd.DataFrame(data)

    return df

def train_decision_trees(X, y):
    #Best Parameters: {'ccp_alpha': 0.2, 'criterion': 'friedman_mse', 'max_depth': 20, 
    #'max_features': 'log2', 'max_leaf_nodes': 5, 'min_impurity_decrease': 0.0, 
    #'min_samples_leaf': 2, 'min_samples_split': 5}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dt = DecisionTreeRegressor(ccp_alpha=0.2, criterion='friedman_mse',max_depth=20,max_features='log2',max_leaf_nodes=5,min_impurity_decrease=0,min_samples_leaf=2,min_samples_split=5)

    scores = []
    r2_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dt.fit(X_train, y_train)

        score = dt.score(X_test, y_test)
        r2 = r2_score(y_test, dt.predict(X_test))
        mae = mean_absolute_error(y_test, dt.predict(X_test))

        scores.append(score)
        r2_scores.append(r2)
        mae_scores.append(mae)

    return dt, np.mean(scores), np.mean(r2_scores), np.mean(mae_scores)

def gridSearch(X, y, dt):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'max_leaf_nodes': [None, 5, 10, 20],
        'min_impurity_decrease': [0.0, 0.1, 0.2],
        'ccp_alpha': [0.0, 0.1, 0.2],
        'criterion': ['mse', 'friedman_mse', 'absolute_error']
    }
    scores = []
    r2_scores = []
    mae_scores = []
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='r2', verbose=0)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        grid_search.fit(X_train, y_train)

        best_dt_model = grid_search.best_estimator_

        score = best_dt_model.score(X_test, y_test)
        r2 = r2_score(y_test, best_dt_model.predict(X_test))
        mae = mean_absolute_error(y_test, best_dt_model.predict(X_test))

        print("Best Parameters:", grid_search.best_params_)
        print("Mean Absolute Error:", mae)
        print("R-squared:", r2)

        scores.append(score)
        r2_scores.append(r2)
        mae_scores.append(mae)
    print("All scores: ", scores)
    print("All r2:", r2_scores)
    print("All mae:", mae_scores)
    return np.mean(scores), np.mean(r2_scores), np.mean(mae_scores)

def main():
    xyz_directory = 'all_structures'
    shifts_directory = 'all_shifts'

    merged_data = load_processed_data(xyz_directory, shifts_directory)

    X = merged_data[['Atom Labels', 'X', 'Y', 'Z']]
    y = merged_data['Shifts']

    dt, accuracy, r2, mae = train_decision_trees(X, y)

    y_pred_mlp = dt.predict(X)

    print("\nDecision Tree:")
    print("Accuracy:", accuracy)
    print("R-squared:", r2)
    print("Mean Absolute Error:", mae)

    
    gs_score, gs_r2, gs_mae = gridSearch(X, y, dt)

    print("\nGrid Search Results:")
    print("Accuracy:", gs_score)
    print("R-squared:", gs_r2)
    print("Mean Absolute Error:", gs_mae)
    


if __name__ == "__main__":
    main()