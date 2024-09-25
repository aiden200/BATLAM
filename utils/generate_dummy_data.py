import os
import numpy as np

def process_npy_files_in_folder(folder_path):
    # Iterate over all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            for filename in os.listdir(os.path.join(root, dir)):

                if filename.endswith(".npy"):
                    file_path = os.path.join(root, dir, filename)
                    
                    # Load the .npy file
                    data = np.load(file_path)
                    print(f"Processing file: {file_path}")

                    # Apply np.tile(x, (2, 1))
                    if data.shape[0] == 4:
                        print("already processed")
                        continue

                    modified_data = np.tile(data, (2, 1))

                    # Save the modified data back to the same file
                    np.save(file_path, modified_data)
                    print(f"File saved: {file_path}")

# Replace 'your_folder_path' with the actual path to your folder
folder_path = 'mp3d_reverb/quad/'

process_npy_files_in_folder(folder_path)
