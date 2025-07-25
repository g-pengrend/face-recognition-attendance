import os
import csv

# Path to your CSV file
csv_file_path = 'AIPD_class.csv'  # Change to your CSV file path

with open(csv_file_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print(reader.fieldnames)
# Optional: Create folders in a specific directory
output_folder = 'students'  # Change if you want a different location
os.makedirs(output_folder, exist_ok=True)

# Read CSV and create folders
with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        serial = row['Serial'].strip()
        name = row['Name'].strip()
        folder_name = f"{serial}_{name}"
        folder_path = os.path.join(output_folder, folder_name)
        
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created: {folder_path}")
        except Exception as e:
            print(f"Failed to create folder '{folder_path}': {e}")
