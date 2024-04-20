import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder

def parse_xyz_file(file_path):
    atoms = []
    coordinates = []
    with open(file_path, 'r') as f:
        num_atoms = int(f.readline())
        _ = f.readline()

        for _ in range(num_atoms):
            atom, x, y, z = f.readline().split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
    
    return np.array(atoms), np.array(coordinates)
'''

def extract_fluorine_data(atoms, coordinates):
    fluorine_indices = [i for i, atom in enumerate(atoms) if atom == 'F']
    fluorine_coordinates = coordinates[fluorine_indices]
    chemical_shift_values = np.random.rand(len(fluorine_indices)) * 10.0  # Random values for demonstration
    fluorine_targets = chemical_shift_values
    
    return fluorine_coordinates, fluorine_targets
'''

def read_chemical_shifts(file_path):
    with open(file_path, 'r') as f:
        chemical_shift_values = [float(line.strip()) for line in f]
    return np.array(chemical_shift_values)

def prepare_data(xyz_file_path, shift_file_path):
    atoms, coordinates = parse_xyz_file(xyz_file_path)
    chemical_shifts = read_chemical_shifts(shift_file_path)
    
    label_encoder = LabelEncoder()
    atom_types_encoded = label_encoder.fit_transform(atoms)
    
    X = np.column_stack((coordinates, atom_types_encoded))
    
    return X, chemical_shifts

def train_svr(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svr = SVR(kernel='rbf')
    svr.fit(X_scaled, y)
    return svr

def predict_chemical_shifts(svr_model, X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    predictions = svr_model.predict(X_scaled)
    return predictions

xyz_file_path = ""#xyz_file_name
shift_file_path = ""#shift_file_name
X, y = prepare_data(xyz_file_path, shift_file_path)
#will probably need extra processing to get all of the xyz's into 1 set (and same for the shifts
svr_model = train_svr(X, y)

predicted_shifts = predict_chemical_shifts(svr_model, X)

'''
print("Predicted chemical shift values for fluorine atoms:")
for i, shift in enumerate(predicted_shifts):
    print(f"Fluorine atom {i+1}: {shift}")
'''
