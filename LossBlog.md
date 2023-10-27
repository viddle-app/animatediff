# Video Generation needs to Lose More

I've been training video models based on Stable Diffusion recently and something dawned on. The loss function used for video generation is the same as the loss function used on image generation, but this is a problem.

One way to think of video is a sequence of images, but I think there is a better way to think about video. Instead of just a sequence of images, think of video as path through the space of images. 

What I am trying to highlight is that with video, we care about how we get from one image to the next. We care about the steps from one image to the next. Is it not just a list of images.

## A Simple Example

To see why this is import let's consider a simple example. Image we have a three frame video with one pixel. To make things even simpler, each pixel value is 128. 

```python
[128, 128, 128]
```

We have to video generation models we are training. The first makes the following estimate for the frames after denoising.

```python
[129, 129, 129]
```

The second model makes the following estimate for the frames.

```python
[127, 129, 127]
```

The first is better. The second model is flickering between the first and third frame. The first model is as smooth as our target. However, both models will have the same loss using the standard per frame loss function. 

The problem is that the loss function looks at each frame independently. It does not care about the path from one frame to the next, but we do. Our brain is very good at detecting interframe changes and this is the key to what makes video interesting. 

## A Better Loss Function

A good loss function needs to capture two characteristics of video, the path from one frame to the next and the appearance of individual frames. 

The current loss function handles the latter, but what about the former?

The easiest thing to do is the compare the frame diffs for the predicted and that target. I thought about how we should combine them, and my guess is adding makes the most sense. 

```python
loss = frame_loss + diff_loss
```

I've consider other options, like frame loss of the first and only diffs, or multiple the individual frames losses times the individual diffs, but adding really captures that we want to optimize both regardless of the other.

It is possible that one loss is more important than the other, so we can add a scaling factor.

```python
loss = frame_loss + diff_loss * diff_scale
```

## More Issues

There is another problem when it comes to interframe changes with Stable Diffusion. 

## Results

Okay so I think this makes sense in theory but does it work in practice? Before even a attempting a test it is worth considering all the reasons it might not work. It might be the case that the easiest optimization path always aligns with the lowest frame diff loss. In that case, the diff loss would be redundant. 

However, if not and the model is free to choose frames with no consideration for the frame diffs, we should be able to detect higher amounts of flickering in the predicted videos.

In other words the temporal spectrum of the predicted videos should be higher than the temporal spectrum of the target videos. So if we take an 1D FFT of each pixel and create a histogram of the magnitudes, we should see more high frequency components in the predicted videos.

