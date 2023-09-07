import numpy as np
import cv2
import os

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

    for i in range(num_frames):
        frame = tensor[i]
        image_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(image_path, frame)
        