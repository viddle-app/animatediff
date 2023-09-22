# Status Report: Tiling AnimateDiff Animations

AnimateDiff has emerged as a promising way to animate Stable Diffusion images consistently. It has a major limitation. The animations must be max 24 frames for V1 based animations and 32 frames for V2 animations.

I've been trying to increase the length of AD animations with some success. You can skip to the end of the what is working, or read along to see how I got there.

## How Stable Diffusion Works

The workhorse of Stable Diffusion is the U-Net. This is where the image generate occurs.

Each time-step the noisy latents travel through the blocks of the U-Net. There are three stages, the down blocks, the middle and the up blocks. Each block applies convolutions, self and cross attention to suss out relationships between the pixels. 

## How AnimateDiff Works

AnimateDiff is different then other txt2vid approaches. It doesn't modify the blocks of the U-Net, but adds a final temporal transformer after each U-Net stage to create consistency across frames. 

I think of it as a similar process to map/reduce. Each frame goes through a U-Net block in isolation, that is the map part. Then all the frames are processed together, that is the reduce part. 

The advantage of this approach is that it doesn't need to modify the U-Net weights. So this tends to preserve the quality of the image generation and allow use to use a variety or Stable Diffusion checkpoints, LoRas and ControlNets unchanged.

However, it has a limitation shared with most video generation methods, it has to generate all the frames at the same time. 

What we need is a way to tile the generation. 

### A First Attempt at Tiling

My first attempt at tiling was to borrow and idea from TEDi. 

TEDi is a method for generating long duration animations with a modified inference and training procedure. 

Typically when generating a batch of frames in a diffusion process, the frames are each generated with the same noise schedule. In TEDi, each frame is at different time-step. One frame is nearly denoised, while the last frame is fully noised, with the frames in between at some time-steps in between.

In this way, after each timestep the first frame is completed and can be popped off the generation queue, and new noisy frame is added. 

The training procedure needs to be modified so model can generate frames with a mix of noise levels. 

It is an autoregressive strategy, which means that errors will accumulate over time. My hope was that the error accumulation would be small enough that it would be acceptable. However, I found that this was not that case, here is an example of a 96 frame animation generated with this method. As you can see it quickly diverges to nothing.

Now something I want to make clear, I didn't follow TEDi by the book. They train with two different types of approaches and use more complex method of including prior information. I think if I did those things it would look better, so I hope someone takes a look at TEDi and tries this again just incase there is a way to make it work. Later in the post I'll explain a way to extend TEDi as well that could make it work even better. 

### Attempt 2: Inpainting

One way to make an image like another is to use inpainting. Inpainting allows you generate a portion of an image that is similar to an existing image. 

The main trick to inpaint is to take an image and add noise to it in accordance with the noise schedule. This will generate a next noisy latent like your input image. 

Each timestep you do this, but here is the trick. Typically, after denoising you would use the resultant less noisy latent. Within painting, you only do this in the region you want to update. 

Every where you throw away the result of denoise, and just replace the current noisy latent with your noised image. 

You're basically using the U-Net to harmonize between the existing image and the portion you are generating. As long as the inpainting region is small enough in comparison to the region you are generating, it will be informed my your noised image.

If you are familiar with inpainting in SD, you probably know there are special inpainting models. These models are not necessary, but like almost all training free approaches, they can be improved with training. So the inpainting work better than vanilla, because of additional training. 

TEDi used this idea in addition to its auto-regressive approach. I was curious if I could get better consistency by windows by doing, and I sort of can. 

If I just pass in one image it doesn't work, but with many images it get's close, but as you can see there is obvious seam. It so close that I can't shake the feeling that there is a bug in my code. Also, maybe there is way to train a model to use this approach? 

### Attempt 3: Shifting Windows

For tiling in the spacial dimension there are a couple approaches. One method is render overlapping windows each time-step and averages. This approach was introduced in MultiDiffusion (and I think Mixture of Diffusers as well). 

Recently, another method has been proposed in the "Any Size Diffusion" which shifts windows between time-steps, essentially allows the U-Net to perform the blending on it's own. 

So I tried this approach and it worked, well at least to a degree. 

It does make one animation that is somewhat consistent. Additiionally we can circularly shift the windows to make the animation loop which is cool. 

However, it makes several different animations that are sticked together of around 24 frames, the max length of the positional encoding. 

However, if we change the offset for the windows we can start to get a single consistent animation, but it is incoherent. As we increase the steps it gets more coherent. 

What we need is a way to guide the generations so it converges faster. 

### Attempt 3: SyncDiffusion

To improve the consistency of MultiDiffusion, SyncDiffusion was introduced. It uses the gradients of the unet to guide the windows to converge to a single animation.

Guidance latents is problematic in Stable Diffusion because they are noisy. To get around this, guidance tends to use the "clean" predicted final latent. 

This works because the U-Net is trained to find noise in a latent, not just for a single step, but what it thinks is the noise for all steps. So the clean final image is just the latent minus the noise modulo some scaling factors.

Since SyncDiffusion uses gradients, it is requires a lot of memory. So much that tiling is only possible with low resolution and small batches. 

Rendering a resolution lower than 512x512 and with windows of less than 16 frames leads to degradation in quality. In the end, it is just not useful as a way to speed up convergence but I didn't get some interesting fails from it.

### Attempt 4: Temporal Reference

"Reference only" is another way guide generation given an example image. It works by using the inpainting trick of noising an input image to the appropriate noise level, then augments the self attention of the U-Net with the features from the input image. 

Could we do the same thing with temporal feature-wise attention in the motion module. 

Essentially we save the prior windows feature in a buffer, then we evaluating the next window they are concatenated on the to the temporal attention. 

It turns out we can and it makes a pretty big difference. 

It is somewhat surprising that this works as well as it does. The motion module adds standard Transformer positional encoding to the features. So when we use the prior window we are saying 

"Generate frame 0 - 24, and here are the prior frames 0 - 24". That is inconsistent but yet it still helps.

### Fine-tune for more winning

With shifting windows and temporal reference we can generate longer fairly consistent animations, but they are jerky and require a lot of steps. The next idea was try to fine-tune the motion module with an augmented training procedure that performs a window shift. 

This means sometimes were are going to execute three steps through the unet. One for the first window, one for the second, and then one at an new timestep for an window that overlaps the prior two. 

Does this make things better? 

The first attempt was not so good. However, I tried again this time lowering the learning rate and randomize the next step. It sort of worked.

It worked in the sense that using this model and the temporal reference only trick we can generate more consistent animations.

Unfornately it was blurry.

### Trying to Understand the Bluriness. 

So at this point we have made some progress with a couple options for improvements so it time to reflect. 

Changing the learning rate helped. That probably means that the new training regime is so different from the prior training method we are "shocking" the model. 

One idea would be to train the motion module from scratch. This way it doesn't have to try to blend between the two competiting training methods. 

When we trained the motion module, we didn't use the temporal reference trick. So maybe we should train it with that trick.

Additionally, if we train temporal reference only we could possible use a larger positional encoding, and then give the frames unique encodings so there would not any encoding inconsistency. 

# Better Positional Encoding

The positional encoding scheme assumed that the features we are currently 






