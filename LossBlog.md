# Video Generation needs to Lose More

I've been training video models based on Stable Diffusion recently and something dawned on. The loss function used for video generation is the same as the loss function used on image generation, but this is a problem.

One way to think of video is a sequence of images, but I think there is a better way to think about video. Instead of just a sequence of images, think of video as path through the space of images. 

What I am trying to highlight is that with video, we care about how we get from one image to the next. We care about the steps from one image to the next. Is it not just a list of images.

## A Simple Example

To see why this is import let's consider a simple example. Imagine we have a three frame video with one pixel. To make things even simpler, each pixel value is 128. 

```python
[128, 128, 128]
```

Say we have two video generation models we are training. The first makes the following estimate for the frames after denoising (in Stable Diffusion we estimate the noise not the image, but just ignore that detail).

```python
[129, 129, 129]
```

The second model makes the following estimate for the frames.

```python
[127, 129, 127]
```

The first is better. The second model is flickering between the first and third frame. The first model is as smooth as our target. However, both models will have the same loss using the standard per frame reconstruction loss function. 

The problem is that the loss function looks at each frame independently. It does not care about the path from one frame to the next, but we do. Our brain is very good at detecting interframe changes and this is the key to what makes video interesting. 

## A Better Loss Function

A good loss function needs to capture two characteristics of video, the path from one frame to the next and the appearance of individual frames. 

The current loss function handles the latter, but what about the former?

The easiest thing to do is the compare the frame diffs for the predicted and that target. I thought about how we should combine them my first guess was to just add them together.

```python
loss = reconstruction_loss + diff_loss
```

Unfortunately this didn't work. Instead of optimizing both losses, during fine-tuning the `diff_loss` loss would get better and the `reconstruction_loss` would get worse.

## A Better Loss Function Take 2

Discrete curves are defined by their initial location in space and the sequences of differences between points. So when comparing two discrete curves one loss could be the L2 of the initial points and the their differences. 

Using this new loss function and continued to monitor the reconstruction loss. This time it continued to improve. 

Here is the python code for the new loss function 

```python
def frame_diff(video_tensor):
    """
    Compute the frame difference for a video tensor.
    video_tensor should have shape (batch_size, channels, frames, height, width)
    """
    return video_tensor[:, :, 1:] - video_tensor[:, :, :-1]

def video_diff_loss(original_video, generated_video):
    # diff both from the first frame
    original_diff = frame_diff(original_video)
    # now compute the frame difference of the generated
    generated_diff = frame_diff(generated_video)

    # add the first frame of the original and the generated
    # to the front of the original_diff and generated_diff
    original_path = torch.cat([original_video[:, :, 0:1], original_diff], dim=2)
    generated_path = torch.cat([generated_video[:, :, 0:1], generated_diff], dim=2)
    # take the difference between the two
    path_difference = original_path - generated_path

    # take the mse loss
    return (path_difference**2).mean()
```

This loss might seem nearly identical to the one before, but there is a notable difference in when we take the mean. Before we took the mean of two differences and added them, here we are taking the mean of one difference. Essentially it is the difference between dividing before adding and the reverse. 

## Results

Okay so I think this makes sense in theory but does it work in practice? Before even a attempting a test it is worth considering all the reasons it might not work. It might be the case that the easiest optimization path always aligns with the lowest frame diff loss. In that case, the diff loss would be redundant. 

However, if not and the model is free to choose frames with no consideration for the frame diffs, we should be able to detect higher amounts of flickering in the predicted videos.

In other words the temporal spectrum of the predicted videos should be higher than the temporal spectrum of the target videos. So if we take an 1D FFT of each pixel and create a histogram of the magnitudes, we should see more high frequency components in the predicted videos.

