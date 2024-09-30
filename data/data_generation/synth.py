import os
import json
import shutil
import numpy as np
from soundspace_sim import SoundspaceSimulator
import sys

base_dir = "/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/binaural"
mp3d_path = "/scratch/ssd1/matterport_habitat/mp3d"
dest_path = "/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/tetra_parallel"

def copy_json_file(src_file, dest_dir):
    file_name = os.path.basename(src_file)
    dest_file = os.path.join(dest_dir, file_name)
    if not os.path.exists(dest_file):
        shutil.copy(src_file, dest_dir)

def process_subdir(subdir_path):
    for file in os.listdir(subdir_path):
        if file.endswith(".json"):
            file_path = os.path.join(subdir_path, file)
            mp3d_room_id = file.replace('.json', '')
            glb_file = os.path.join(os.path.join(mp3d_path, mp3d_room_id), file.replace('.json', '.glb'))
            rirs_path = os.path.join(dest_path, mp3d_room_id)  # directory to save generated RIRs for mp3d_id
            os.makedirs(rirs_path, exist_ok=True)
            copy_json_file(file_path, rirs_path)  # copy json file to the new directory
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    audio_sensor = value["audio_sensor"]
                    source = value["source"]

                    print(f"Generating data for - File_ID: {key}, GLB: {glb_file}")

                    rir_path_dest = os.path.join(rirs_path, f"{key}.npy")  # destination RIR filepath
                    sensor_position = np.array([audio_sensor[0], -1 * audio_sensor[2], audio_sensor[1]])
                    source_position = [np.array([source[0], -1 * source[2], source[1]])]
                    soundsim = SoundspaceSimulator(glb_file, rir_path_dest, mic_pos=sensor_position, sources_pos=source_position)
                    soundsim.generate_rir_data()

def main():
    if len(sys.argv) < 2:
        print("Please provide a file with subdirectories to process.")
        return

    sublist_file = sys.argv[1]
    
    # Read subdirectories from the provided file
    with open(sublist_file, 'r') as f:
        subdirs = f.read().split()
    
    for subdir in subdirs:
        process_subdir(subdir)

if __name__ == "__main__":
    main()

