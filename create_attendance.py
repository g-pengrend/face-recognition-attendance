import csv
import os

# Path to your CSV file
csv_file = "CI2501A_attd_list.csv"

# Folder where all subfolders will be created
output_dir = "output_folders"
os.makedirs(output_dir, exist_ok=True)

# Read CSV and create folders
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:  # Ensure at least two columns: serial, name
            serial = row[0].strip()
            name = row[1].strip()
            folder_name = f"{serial}_{name}".replace(" ", "_")  # Replace spaces for neatness
            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created: {folder_path}")

print("All folders created successfully.")
