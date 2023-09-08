import glob
import numpy as np
import cv2
import os
from PIL import Image

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

if __name__ == "__main__":
    images = glob.glob("images/*.png")
    images.sort()
    create_gif(images, "output.gif", duration=1000/15, loop=0)