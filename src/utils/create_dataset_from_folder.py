import os
from datasets import Dataset, DatasetDict
import pandas as pd

def create_video_dataset(folder_path):
    # List to store video file paths
    video_paths = []

    # Supported video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    # Scan the folder for video files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in video_extensions):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)

    # Create a DataFrame with video paths
    df = pd.DataFrame(video_paths, columns=['video_path'])

    # Convert DataFrame to Dataset
    dataset = Dataset.from_pandas(df)

    # Convert to DatasetDict and save to disk
    new_dataset_dict = DatasetDict({'train': dataset})
    new_dataset_dict.save_to_disk("video_dataset")

    return new_dataset_dict

if __name__ == "__main__":
  create_video_dataset("data/videos")