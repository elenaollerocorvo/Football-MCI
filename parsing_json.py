# Script to process and clean label JSON files
# Removes unnecessary fields and normalizes data into a DataFrame

# Import libraries
import pandas as pd  # For data manipulation
import json  # For reading/writing JSON
import os  # For file system operations

# Path to the folder with label JSON files
folder_path = "C:\\Users\\Domagoj\\Desktop\\Session 2 json"

# List to store data from all JSONs
json_data_list = []

def rectify_json():
    """
    Removes unnecessary fields from all JSON files in the folder.
    The removed fields are: project_name, labeler, method, decision_basis
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Read JSON
        f = open(file_path, 'r')
        data = json.load(f)
        
        # Remove unnecessary fields for further processing
        # Note: Using data.pop('key', None) is safer here to avoid KeyErrors
        del data['project_name']
        del data['labeler']
        del data['method']
        del data['decision_basis']
        f.close()
        
        # Save modified JSON
        f = open(file_path, 'w')
        json.dump(data, f, indent=4)


# Load all JSON files into a list
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path) as f:
            data = json.load(f)
            
            # If there are no touches in frames, also clear the milliseconds field
            if data['button_presses'] == '':
                data['button_presses_ms'] = ''
                print(data['button_presses_ms'])
            
            json_data_list.append(data)

# Execute cleaning function
rectify_json()

# Normalize JSON list to pandas DataFrame
cijela_lista = pd.json_normalize(json_data_list)

# Configure pandas to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
