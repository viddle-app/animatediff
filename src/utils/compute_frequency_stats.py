import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import csv

def load_images(folder_path):
    images = []
    file_names = sorted(os.listdir(folder_path))  # Sort the file names
    for file_name in file_names:
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            images.append(image)
    return images

def preprocess_images(images):
    # Convert images to tensors and stack them
    transform = transforms.ToTensor()
    tensors = [transform(img) for img in images]
    stacked_tensors = torch.stack(tensors, dim=0)  # Shape: [num_frames, channels, height, width]
    return stacked_tensors

def compute_1d_fft(tensor):
    fft_result = torch.fft.fft(tensor, dim=0)  # Perform FFT along the frames dimension
    return fft_result

def calculate_frequency_spectrums(fft_result):
    spectrums = torch.abs(fft_result)
    return spectrums

def aggregate_statistics(spectrums):
    # Reshape spectrums to merge channel, height, and width dimensions
    reshaped_spectrums = spectrums.view(spectrums.size(0), -1)

    mean_spectrum = torch.mean(reshaped_spectrums, dim=-1)
    min_spectrum, _ = torch.min(reshaped_spectrums, dim=-1)
    max_spectrum, _ = torch.max(reshaped_spectrums, dim=-1)

    return mean_spectrum, min_spectrum, max_spectrum

def save_to_csv(mean_spectrum, min_spectrum, max_spectrum, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frequency', 'Mean', 'Min', 'Max'])

        # Iterate over the frequencies and write the results
        for i in range(len(mean_spectrum)):
            csvwriter.writerow([i, mean_spectrum[i].item(), min_spectrum[i].item(), max_spectrum[i].item()])

def process_folder(folder_path):
    images = load_images(folder_path)
    tensor_images = preprocess_images(images)
    fft_result = compute_1d_fft(tensor_images)
    frequency_spectrums = calculate_frequency_spectrums(fft_result)
    return aggregate_statistics(frequency_spectrums)

def aggregate_across_folders(folder_paths):
    sum_means = None
    overall_min = None
    overall_max = None
    count = 0

    for folder_path in folder_paths:
        mean_spectrum, min_spectrum, max_spectrum = process_folder(folder_path)
        sum_means = mean_spectrum if sum_means is None else sum_means + mean_spectrum
        overall_min = min_spectrum if overall_min is None else torch.min(overall_min, min_spectrum)
        overall_max = max_spectrum if overall_max is None else torch.max(overall_max, max_spectrum)
        count += 1

    average_mean = sum_means / count
    return average_mean, overall_min, overall_max

def process_and_save(folder_group, csv_file_path):
    avg_mean_spectrum, overall_min_spectrum, overall_max_spectrum = aggregate_across_folders(folder_group)
    save_to_csv(avg_mean_spectrum, overall_min_spectrum, overall_max_spectrum, csv_file_path)
    print(f"Aggregate results for {len(folder_group)} folders saved to", csv_file_path)

def main(root_folder1, csv_path1, root_folder2, csv_path2):
    folder_group1 = [os.path.join(root_folder1, folder) for folder in os.listdir(root_folder1) if os.path.isdir(os.path.join(root_folder1, folder))]
    folder_group2 = [os.path.join(root_folder2, folder) for folder in os.listdir(root_folder2) if os.path.isdir(os.path.join(root_folder2, folder))]

    process_and_save(folder_group1, csv_path1)
    process_and_save(folder_group2, csv_path2)

if __name__ == "__main__":
    # Specify the root folders and corresponding CSV file paths
    root_folder1 = "/mnt/newdrive/viddle-animatediff/fft-comparisons/diff_mid"
    csv_file_path1 = "/mnt/newdrive/viddle-animatediff/output/diff_mid_aggregate_frequency_stats.csv"

    root_folder2 = "/mnt/newdrive/viddle-animatediff/fft-comparisons/recon_mid"
    csv_file_path2 = "/mnt/newdrive/viddle-animatediff/output/recon_mid_aggregate_frequency_stats.csv"

    main(root_folder1, csv_file_path1, root_folder2, csv_file_path2)


  
