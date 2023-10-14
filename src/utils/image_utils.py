import glob
import numpy as np
import cv2
import os
from PIL import Image
import torch
import imageio


def tensor_to_image_sequence(tensor, output_dir):
    """
    Convert a tensor of frames to a sequence of images.
    Parameters:
    - tensor: The tensor of frames with shape (num_frames, height, width, channels).
    - output_dir: Directory to save the output images.
    """
    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    tensor = tensor.cpu().permute(1, 2, 3, 0).numpy()

    if np.isnan(tensor).any():
      print("Tensor contains NaN values.")


    # Get the shape details from the tensor
    num_frames, height, width, channels = tensor.shape

    # Convert from 0 to 1 to 0 to 255
    tensor = tensor * 255

    # Ensure the tensor values are in the correct range [0, 255]
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)

    # swap the channels from RGB to BGR
    tensor = tensor[...,::-1]

    for i in range(num_frames):
        frame = tensor[i]
        image_path = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(image_path, frame)

def save_gif_from_tensor(tensor, filename):
    """
    Save a gif from a tensor.
    
    :param tensor: a (F, C, H, W) shaped torch.Tensor. 
                   B = batch size, C = num channels, 
                   F = num frames, H = image height, W = image width
    :param filename: path and filename to save the gif
    """
    # Ensure the tensor is on the CPU
    tensor = tensor.cpu()
    
    # Convert the PyTorch tensor to NumPy array and 
    # transpose it to (F, H, W, C) for imageio
    gif_np = tensor.numpy().transpose(0, 2, 3, 1)
    
    # Normalize the gif to be between 0 and 255 and convert type to uint8
    gif_np = gif_np * 255
    gif_np = gif_np.astype('uint8')
    
    # Save the gif
    imageio.mimwrite(filename, [f for f in gif_np], duration=1/8, loop=0)

def create_gif(image_files, output_file, duration=500, loop=0):
    """
    Creates a looping GIF from a sequence of images.
    
    Parameters:
        image_files (list of str): List of paths to the image files.
        output_file (str): Path to save the output GIF.
        duration (int): Duration each frame lasts in the GIF in milliseconds. Default is 500ms.
        loop (int): Number of times the GIF should loop. 0 for infinite looping.
    """
    
    # Load all the images using PIL
    images = [Image.open(image_file) for image_file in image_files]
    
    # Save the sequence as a GIF
    images[0].save(output_file, save_all=True, 
                   append_images=images[1:], 
                   loop=loop, 
                   duration=duration, 
                   optimize=False, quality=100)
    print(f"Saved gif to {output_file}")

def create_mp4_from_images(images_folder, output_path, fps=15):
    cmd = f"ffmpeg -r {fps} -i {images_folder}/%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {output_path}"
    os.system(cmd)

def compute_spectrum(input_seq):
    # Assume input_seq is of shape [16, H, W, 3]
    # Perform FFT along the first dimension (time dimension)
    input_seq = input_seq.permute(1, 2, 3, 0)
    print("Input shape:", input_seq.shape)
    spectrum = torch.fft.fft(input_seq, dim=0)

    print("Spectrum shape:", spectrum.shape)
    
    # Compute the magnitude
    magnitude = torch.abs(spectrum)
    
    # Reorder dimensions so we get [H, W, 3, 16] and take log-scale for visualization
    magnitude = magnitude.permute(1, 2, 3, 0)
    magnitude = torch.log(1 + magnitude)
    
    # Normalize the magnitude values to [0, 1] range for visualization
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    return magnitude

# TODO normalize all the outputs
def save_spectrum_images(magnitude, output_folder):
    magnitude = magnitude.permute(3, 0, 1, 2)
    # Ensure the directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Making sure magnitude is in [0, 1] range
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    
    # Iterating over the first dimension (frequencies) and saving images
    for i in range(min(8, magnitude.shape[0])):
        # Extracting i-th frame from the magnitude tensor
        img = magnitude[i]
        
        # Rescale to [0, 255] and convert to uint8
        img_np = (img.numpy() * 255).astype(np.uint8)
        
        # Ensure that array shape is (H, W, C)
        img_np = np.squeeze(img_np)
        
        # Save the image using Pillow
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(output_folder, f'spectrum_{i}.png'))

def save_statistics(magnitude, output_folder):
    _, _, _, C = magnitude.shape
    stats = []
    for i in range(min(8, C//2)):
        band = magnitude[..., 2*i:2*i+3]
        mean = torch.mean(band)
        std_dev = torch.std(band)
        stats.append((mean.item(), std_dev.item()))
    # Save the statistics into a file
    with open(os.path.join(output_folder, 'statistics.txt'), 'w') as file:
        for i, (mean, std_dev) in enumerate(stats):
            file.write(f'Frequency {i}:\n Mean: {mean:.6f}, Standard Deviation: {std_dev:.6f}\n\n')
        
def load_video_as_tensor(video_path, skip_frame=4, max_frames=None):
    """
    Load video and convert it to a PyTorch tensor.

    Parameters:
    - video_path (str): Path to the video file.
    - skip_frame (int): Number of frames to skip.
    - max_frames (int or None): Max number of frames to load.

    Returns:
    torch.Tensor: Tensor with shape (B, C, F, H, W).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break
        
        # Skip frames as per `skip_frame` value
        if frame_count % skip_frame != 0:
            frame_count += 1
            continue
        
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize pixel values from [0, 255] to [-1, 1]
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1

        frames.append(frame_tensor)
        frame_count += 1
    
    cap.release()

    # Stack frames into a single tensor
    video_tensor = torch.stack(frames)

    # Add batch dimension and reorder the dimensions to (B, C, F, H, W)
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)

    return video_tensor

if __name__ == "__main__":
    fps = 24
    create_mp4_from_images("images", "output.mp4", fps=fps)
    images = glob.glob("images/*.png")
    images.sort()
    create_gif(images, "output.gif", duration=1000/fps, loop=0)