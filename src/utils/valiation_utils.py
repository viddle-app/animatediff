import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import ToPILImage
import json
import matplotlib.pyplot as plt

# Instantiate the transformation
to_pil_image = ToPILImage()

def concatenate_images(img1, img2, img3):
    width, height = img1.width + img2.width + img3.width, max(img1.height, img2.height, img3.height)
    new_img = Image.new('RGB', (width, height))
    
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (img1.width + img2.width, 0))
    
    return new_img

def concatenate_images(img1, img2, img3):
    """
    Concatenate three PIL Images horizontally.
    
    Params:
        img1, img2, img3 (PIL.Image.Image): Images to be concatenated
    
    Returns:
        PIL.Image.Image: Concatenated image
    """
    width, height = img1.size
    
    concatenated_img = Image.new('RGB', (3*width, height))
    concatenated_img.paste(img1, (0, 0))
    concatenated_img.paste(img2, (width, 0))
    concatenated_img.paste(img3, (2*width, 0))
    
    return concatenated_img

def save_images(image_sequence, folder):
    """
    Save a sequence of images to a specified folder.
    
    Params:
        image_sequence (list of PIL.Image.Image): Sequence of images to be saved
        folder (str): Path to the folder to save the images
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for i, img in enumerate(image_sequence):
        img.save(os.path.join(folder, f"frame_{i:04d}.png"))

def create_gif_from_images(image_sequence, output_gif_path, fps=8):
    """
    Create a GIF from a sequence of images.
    
    Params:
        image_sequence (list of PIL.Image.Image): Sequence of images to be used in the GIF
        output_gif_path (str): Path to save the resulting GIF
        fps (int, optional): Frames per second for the GIF. Defaults to 8.
    """
    image_sequence[0].save(output_gif_path, 
                           save_all=True, 
                           append_images=image_sequence[1:], 
                           loop=0, 
                           duration=1000//fps)


def compute_3d_fft(tensor):
    """
    Compute 3D FFT of a tensor and return the magnitude and phase.
    
    Params:
        tensor (torch.Tensor): 4D tensor (channel, time, height, width)
        
    Returns:
        magnitude (torch.Tensor): Magnitude of FFT
        phase (torch.Tensor): Phase of FFT
    """
    # FFT along the time, height, and width dimensions
    fft_result = torch.fft.fftn(tensor, dim=(1, 2, 3))

    # Shift the FFT result
    fft_shifted = torch.fft.fftshift(fft_result, dim=(1, 2, 3))

    # Calculate the magnitude and phase
    magnitude = torch.sqrt(fft_shifted.real**2 + fft_shifted.imag**2)
    phase = torch.atan2(fft_shifted.imag, fft_shifted.real)

    return magnitude, phase

def compute_2d_fft_batch(tensor):
    """
    Compute 2D FFT of a batch of 2D images/tensors and return the magnitude and phase.
    
    Params:
        tensor (torch.Tensor): 4D tensor (batch_size, channel, height, width)
        
    Returns:
        magnitude (torch.Tensor): Magnitude of FFT for each image in the batch
        phase (torch.Tensor): Phase of FFT for each image in the batch
    """
    # FFT along the height and width dimensions
    fft_result = torch.fft.fftn(tensor, dim=(2, 3))

    # Shift the FFT result
    fft_shifted = torch.fft.fftshift(fft_result, dim=(2, 3))

    # Calculate the magnitude and phase
    magnitude = torch.sqrt(fft_shifted.real**2 + fft_shifted.imag**2)
    phase = torch.atan2(fft_shifted.imag, fft_shifted.real)

    return magnitude, phase

def compute_luminance_histogram(frames: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    """
    Compute the luminance histogram for a tensor of image frames.
    
    Parameters:
        frames (torch.Tensor): a tensor of image frames of shape (num_frames, height, width, channels).
        num_bins (int): the number of bins in the histogram.
        
    Returns:
        torch.Tensor: a tensor containing the luminance histogram for each frame.
    """
    # Ensure the input is on the proper device and type
    frames = frames.to(torch.float32)
    
    # Check if the input has the correct shape
    if len(frames.shape) != 4 or frames.shape[-1] != 3:
        raise ValueError("Input tensor should have shape (num_frames, height, width, 3). But had", frames.shape)
    
    # Convert RGB to luminance using the standard NTSC conversion formula.
    # L = 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299*frames[..., 0] + 0.587*frames[..., 1] + 0.114*frames[..., 2]
    
    # Normalize the luminance to be between 0 and 1
    luminance = (luminance - torch.min(luminance)) / (torch.max(luminance) - torch.min(luminance))
    
    # Initialize tensor to hold histogram data
    histograms = torch.zeros((frames.shape[0], num_bins), device=frames.device, dtype=torch.float32)
    
    # Compute histogram for each frame
    for i in range(frames.shape[0]):
        histograms[i]  = torch.histc(luminance[i].flatten(), bins=num_bins, min=0, max=1)
    
    return histograms

def plot_histograms(histograms: torch.Tensor, bin_edges = torch.linspace(0, 1, steps=257)) -> torch.Tensor:
    """
    Plot histograms and convert the plots to a tensor of images.

    Parameters:
        histograms (torch.Tensor): a tensor containing histograms of shape (num_frames, num_bins).
        bin_edges (torch.Tensor): a tensor containing the bin edges.

    Returns:
        torch.Tensor: a 4D tensor containing RGB images of all the plots.
    """
    # Ensure CPU and NumPy compatibility
    histograms = histograms.cpu().numpy()
    bin_edges = bin_edges.cpu().numpy()

    # Initialize a list to store image arrays
    images = []

    # Loop through all histograms and generate plot images
    for hist in histograms:
        # Creating a figure and axis object
        fig, ax = plt.subplots()
        
        # Plotting the histogram
        ax.plot(bin_edges[:-1], hist, lw=2)
        
        # Convert the Matplotlib figure to a NumPy array
        fig.canvas.draw()
        img_arr = np.array(fig.canvas.renderer.buffer_rgba())
        
        # Appending to images list
        images.append(img_arr)
        
        # Close the figure
        plt.close(fig)

    # Converting the list of image arrays to a 4D tensor
    image_tensor = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)

    return image_tensor

def compute_1d_spectrum(input_seq):
    # Assume input_seq is of shape [16, H, W, 3]
    # Perform FFT along the first dimension (time dimension)
    print("Input shape:", input_seq.shape)
    spectrum = torch.fft.fft(input_seq, dim=0)

    print("Spectrum shape:", spectrum.shape)
    
    # Compute the magnitude
    magnitude = torch.abs(spectrum)

    return magnitude

def plot_sum_of_pixel_values(frames_tensor, save_path='pixel_sum_plot.png'):
    """
    This function takes a PyTorch tensor representing a set of image frames, 
    sums the pixel values in each frame, and then saves the resulting plot as an image file.

    Parameters:
    - frames_tensor: A 4D PyTorch tensor with shape [num_frames, height, width, num_channels]
    - save_path: String, path where the plot image will be saved.
    """
    
    # Ensure that the input tensor has 4 dimensions
    if len(frames_tensor.shape) != 4:
        raise ValueError("Input tensor must have 4 dimensions: [num_frames, height, width, num_channels]")
    
    # Calculate the sum of pixel values for each frame
    pixel_sums = frames_tensor.sum(dim=(1,2,3)).numpy()

    # Create a line plot of the pixel sums
    plt.plot(pixel_sums)
    plt.xlabel('Frame')
    plt.ylabel('Total Pixel Value')
    plt.title('Sum of Pixel Values in Each Frame')
    
    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()

def process_image_sequence(input_folder, output_folder):
    """
    Load image sequence, compute 3D FFT, and save magnitude and phase spectra as images.
    
    Params:
        input_folder (str): Path to folder containing image sequence
        mag_output_folder (str): Path to save magnitude images
        phase_output_folder (str): Path to save phase images
    """
    fft_images_folder = os.path.join(output_folder, "fft_images")
    fft_gif_path = os.path.join(output_folder, "fft.gif")

    image_files = sorted(os.listdir(input_folder))
    
    # Assume images are RGB and have the same dimensions
    first_image = Image.open(os.path.join(input_folder, image_files[0]))
    width, height = first_image.size
    
    # Initialize a 4D tensor to hold the image sequence (channel, time, height, width)
    image_sequence = torch.empty(3, len(image_files), height, width)

    # Define a transformation: ToTensor converts PIL Image or numpy.ndarray (H x W x C) to torch.FloatTensor of shape (C x H x W)
    transform = transforms.ToTensor()
    
    # Load images into the tensor
    for t, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path).convert("RGB")  # Ensure RGB even if input images vary
        image_sequence[:, t, :, :] = transform(image)
    
    # Compute FFT magnitude and phase
    magnitude, phase = compute_2d_fft_batch(image_sequence)
    
    # Normalizing magnitude to [0, 1]
    magnitude = torch.log1p(magnitude)
    max_val = torch.max(magnitude)
    min_val = torch.min(magnitude)
    print("min_val:", min_val)
    print("max_val:", max_val)
    magnitude = (magnitude - min_val) / (max_val - min_val)

    # Normalize the phase from -pi to pi to [0, 1]
    phase = (phase + np.pi) / (2 * np.pi)

    # Make sure the output directories exist
    os.makedirs(fft_images_folder, exist_ok=True)

    magnitude = magnitude.permute(1, 0, 2, 3)
    _phase = phase.permute(1, 0, 2, 3)
    image_sequence = image_sequence.permute(1, 0, 2, 3)

    permuted_pixels = image_sequence.permute(0, 2, 3, 1)

    lumancity = plot_histograms(compute_luminance_histogram(permuted_pixels))

    plot_sum_of_pixel_values(permuted_pixels, os.path.join(output_folder, "pixel_sum_plot.png"))
    
    # save a gif of all three magnitude, phase, and original images tensors
    concatenated_images = [
      concatenate_images(to_pil_image(lumancity[t]), 
                       to_pil_image(magnitude[t]), 
                       to_pil_image(image_sequence[t]))
        for t in range(image_sequence.shape[0])
    ]

    # Save the concatenated images to a folder
    save_images(concatenated_images, fft_images_folder)

    # Create a GIF from the concatenated image sequence
    create_gif_from_images(concatenated_images, fft_gif_path)

    spectrum_1d_folder = os.path.join(output_folder, "spectrum_1d")

    os.makedirs(spectrum_1d_folder, exist_ok=True)


    spectrum1d = compute_1d_spectrum(image_sequence.permute(0, 2, 3, 1))
    # for each frequency band in the spectrum1d save an image of the magnitude normalized for that frame
    # Iterating over the first dimension (frequencies) and saving images

    frequency_energies = []

    for i in range(min(8, spectrum1d.shape[0])):
        # Extracting i-th frame from the magnitude tensor
        img = spectrum1d[i]

        # take the logp1 of the magnitude
        img = torch.log1p(img)

        frequency_energies.append(img.sum().item())

        #normalize 
        img = (img - img.min()) / (img.max() - img.min())
        
        # Rescale to [0, 255] and convert to uint8
        img_np = (img.numpy() * 255).astype(np.uint8)
        
        # Ensure that array shape is (H, W, C)
        img_np = np.squeeze(img_np)

        
        
        # Save the image using Pillow
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(spectrum_1d_folder, f'spectrum_{i}.png'))


    # save the temporal total magnitudes and relative magnitudes
    temporal_total_magnitudes = torch.sum(magnitude, dim=(2, 3))
    temporal_total_magnitudes_ratio = temporal_total_magnitudes / temporal_total_magnitudes[0]

    # save the entropy of the log normalized spectrum
    spectrum_entropy = -torch.sum(magnitude * torch.log(magnitude), dim=(2, 3))

    # save into a stats.json
    stats = {
        "temporal_total_magnitudes": temporal_total_magnitudes.tolist(),
        "temporal_total_magnitudes_ratio": temporal_total_magnitudes_ratio.tolist(),
        "temporal_frequency_energies": frequency_energies,
        # "spectrum_entropy": spectrum_entropy.tolist(),
    }

    with open(os.path.join(output_folder, "stats.json"), "w") as f:
        json.dump(stats, f)

# for validation of during training
# I want to graph the temporal frequency energy differences
# the video difference


# write a function to load to stats.json files and compare the stats
def compare_stats(stats_file_0, stats_file_1):
    with open(stats_file_0, "r") as f:
        stats_0 = json.load(f)

    with open(stats_file_1, "r") as f:
        stats_1 = json.load(f)

    # compare the temporal total magnitudes
    temporal_total_magnitudes_0 = np.array(stats_0["temporal_total_magnitudes"])
    temporal_total_magnitudes_1 = np.array(stats_1["temporal_total_magnitudes"])

    # get the ratios between temporal_total_magnitudes_0 and temporal_total_magnitudes_1
    temporal_total_magnitudes_compare = temporal_total_magnitudes_0 / temporal_total_magnitudes_1

    temporal_total_magnitudes_ratio_0 = np.array(stats_0["temporal_total_magnitudes_ratio"])
    temporal_total_magnitudes_ratio_1 = np.array(stats_1["temporal_total_magnitudes_ratio"])

    # get the ratios between temporal_total_magnitudes_0 and temporal_total_magnitudes_1
    temporal_total_magnitudes_ratio_compare = temporal_total_magnitudes_ratio_0 / temporal_total_magnitudes_ratio_1

    return temporal_total_magnitudes_compare, temporal_total_magnitudes_ratio_compare

def graph_temporal_frequencies(stats_file_0, stats_file_1):
    with open(stats_file_0, "r") as f:
        stats_0 = json.load(f)

    with open(stats_file_1, "r") as f:
        stats_1 = json.load(f)

    temporal_frequency_energies_0 = np.array(stats_0["temporal_frequency_energies"])
    temporal_frequency_energies_1 = np.array(stats_1["temporal_frequency_energies"])

    plt.plot(temporal_frequency_energies_0, label="temporal_frequency_energies_0")
    plt.plot(temporal_frequency_energies_1, label="temporal_frequency_energies_1")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # the_id = "e428b5a8-0ead-46bf-9894-0c594f773366"
    # the_id_1 = "475b254b-8361-48b7-9173-ac2eb967aa6d"
    the_id = "989a1946-7476-45a7-ab66-cc1530db4d92"
    
    # output_folder_0 = f"validation/{the_id}/stats.json"
    # output_folder_1 = f"validation/{the_id_1}/stats.json"
    # results = compare_stats(output_folder_0, output_folder_1)
    # print(results)
    old_id = "a4ac5f05-8df5-43b5-b571-6c478355db1d"
    input_folder = f"output/{old_id}"
    old_folder = f"validation/{old_id}"
    process_image_sequence(input_folder, old_folder)
    