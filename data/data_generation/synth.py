import os
import json
import shutil
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from soundspace_sim import SoundspaceSimulator 

base_dir = "/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/binaural"
mp3d_path = "/scratch/ssd1/matterport_habitat/mp3d"
dest_path = "/scratch/ssd1/audio_datasets/SpatialSounds/mp3d_reverb/tetra"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_json_file(src_file, dest_dir):
    file_name = os.path.basename(src_file)
    dest_file = os.path.join(dest_dir, file_name)
    if not os.path.exists(dest_file):
        shutil.copy(src_file, dest_dir)

def process_key(key, value, glb_file, rirs_path):
    try:
        audio_sensor = value["audio_sensor"]
        source = value["source"]
        
        logging.info(f"Generating data for - File_ID: {key}, GLB: {glb_file}")
        
        rir_path_dest = os.path.join(rirs_path, f"{key}.npy")
        sensor_position = np.array([audio_sensor[0], -1*audio_sensor[2], audio_sensor[1]]) 
        source_position = [np.array([source[0], -1*source[2], source[1]])]
        soundsim = SoundspaceSimulator(glb_file, rir_path_dest, mic_pos=sensor_position, sources_pos=source_position)
        soundsim.generate_rir_data()
        logging.info(f"Completed processing for - File_ID: {key}")
    except Exception as e:
        logging.error(f"Error processing key {key}: {str(e)}")

def process_file(file_path, mp3d_room_id):
    try:
        glb_file = os.path.join(os.path.join(mp3d_path, mp3d_room_id), mp3d_room_id + '.glb')
        rirs_path = os.path.join(dest_path, mp3d_room_id)
        os.makedirs(rirs_path, exist_ok=True)
        copy_json_file(file_path, rirs_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_key, key, value, glb_file, rirs_path): key for key, value in data.items()}
            
            for future in as_completed(futures, timeout=600):  # 10 minutes timeout
                key = futures[future]
                try:
                    future.result(timeout=60)  # 60 seconds timeout for each key
                except TimeoutError:
                    logging.error(f"Timeout occurred while processing key {key}")
                except Exception as e:
                    logging.error(f"Error occurred while processing key {key}: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")

def main():
    start_time = time.time()
    processed_files = 0
    
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        if os.path.isdir(subdir_path):
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for file in os.listdir(subdir_path):
                    if file.endswith(".json"):
                        file_path = os.path.join(subdir_path, file)
                        mp3d_room_id = file.replace('.json', '')
                        futures.append(executor.submit(process_file, file_path, mp3d_room_id))
                
                for future in as_completed(futures):
                    try:
                        future.result(timeout=1800)  # 30 minutes timeout for each file
                        processed_files += 1
                        elapsed_time = time.time() - start_time
                        logging.info(f"Processed {processed_files} files. Elapsed time: {elapsed_time:.2f} seconds")
                    except TimeoutError:
                        logging.error("Timeout occurred while processing a file")
                    except Exception as e:
                        logging.error(f"Error occurred while processing a file: {str(e)}")

if __name__ == "__main__":
    main()
