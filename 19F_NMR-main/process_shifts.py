import os

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
    all_shifts.append(numbers)

all_shifts = list()
directory = "all_shifts"
for file in os.listdir(directory):
    extract_shifts(os.path.join(directory, file))
print(len(all_shifts))