# Longer AnimateDiff Animations Part Two: Overlapping Windows

In the first blog post of this series I talked about how we can modify the temporal attention of the motion module to use features from the prior window. This helped, but it was far from perfect, even with fine-tuning.

Luckily there is another technique we can use to help the motion module. We can use overlapping windows.

There are many ways we can overlap the windows, and luckily some prior art in the spatial realm we can borrow. 

## Spatial Tiling

Often when generating large images we would like to tile the generation because the we can only generate so many pixels due to memory constraints. 

However, we need make sure that there are no seams between the tiles. We need some way to blend the tiles together.

The simplest way to do this is to just average the latents of a shared overlapping region. This is the approach used in MultiDiffusion and Mixture of Diffusers.

The downside of this approach is that it is not very efficient. We are generating the same pixels multiple times.

Another method for tiling is to shift the non-overlapping windows between time-steps. Although for a given time-step there is no overlapping, which would create seams, because we shift the windows each timestep, the seams are average out. This is the approach used in "Any Size Diffusion".

Where as the MultiDiffusion approaches averages latents each time-step, the "Any Size Diffusion" approach let's the U-Net harmonize the latents over time. This is more efficent and doesn't require generating the same pixels multiple times in a single time-step. However, it can result in having to use more time-steps to get the same quality.

In other case we shouldn't be dissappointed if we end up having to compute more. Trading memory for compute is often a good trade-off, especially if the alternative is not being able to generate the image at all.

## Temporal Tiling

So we can use the same approaches for temporal tiling. We can either average the latents of the overlapping windows, or we can shift the windows between time-steps.

### Averaging

For average we have a few parameters to tweak, specifically the window size and the overlap size. AFAICT, AnimateDiff doesn't work well with a window size is small, like under 12 frames, so we will try to keep that as large as we can. 

The overlap is something we can tweak. The larger the overlap the more blending we get, but the more compute we need. However, in theory at least, we should get more consistency with larger overlaps. 

Here is the maximum overlap, where we shift the window by one frame each time-step.

![Maximum Overlap](images/overlap-max.gif)

Not good. We averaged out all the detail and the motion. 

Let's try an overlap of 12 frames.

![Overlap 12](images/overlap-12.gif)

Better.

Now let's try using the temporal reference only from the last blog post.

![Overlap 12 Temporal Reference](images/overlap-12-temporal-reference.gif)

Better still.

With the custom model we trained for reference only.

![Overlap 12 Temporal Reference Fine-tuned](images/overlap-12-temporal-reference-fine-tuned.gif)

Betterer

### Shifting Between Time-Steps

Now let's try shifting windows between time-steps. I'm going to change things slightly, and we are going to circular shift the windows. This will make animation loop. 

Again we can play with the window size and the amount we will shift. First let's try a 1 frame shift.

![Shift 1](images/shift-1.gif)

Not good, but better than average. We have a seamless animation, but it consists of two obvious different scenes.

Let's try a 2 frame shift.

![Shift 2](images/shift-2.gif)

Weird. It is more of one consistent scene but it the world is on acid. 

Let's increase the timesteps from 20 to 40. 

![Shift 2 40 Frames](images/shift-2-40.gif)

Better. Let's go higher. 160.

![Shift 2 160 Frames](images/shift-2-160.gif)

Okay, pretty good it is converging. 

I've played around with other values and between 2-4 is where you get the best results but you need a lot of samples to make it work well. 

Continueing our abalation we will add the temporal reference only trick.

![Shift 2 20 Steps Temporal Reference](images/shift-2-20-temporal-reference.gif)

Now if the custom model we trained for reference only.

![Shift 2 20 Steps Temporal Reference Fine-tuned](images/shift-2-20-temporal-reference-fine-tuned.gif)

Better better. 

Alright lets try it without looping

![Shift 2 20 Steps Temporal Reference Fine-tuned No Loop](images/shift-2-20-temporal-reference-fine-tuned-no-loop.gif)

Oh no. It looks terrible. 

Well I knew this would happen. The problem is with non-overlapping windows we have some windows that are 1 - window length long. AD can't animate in a small frames and it leads to noise. This is messing up the final image. So the shifting approach only works naively with looping.

If we don't want the animation to loop, we can't use shifting by itself. Let's mix average and shifting. We will also ways have a window length frames at the beginning and end, and average those.

![Shift 2 20 Steps Temporal Reference Fine-tuned No Loop](images/shift-2-20-temporal-reference-fine-tuned-no-loop-mix.gif)

Better-ish.

### Fine-tuning

The shifting approach was neat, but it didn't result in the promised effiency gains. The model is just not trained to harmonize the latents like that, so it needed more steps to correct errors.

So let's try fine-tuning the motion module with a modified training procedure so it learns how to mix to adjancent windows.

This will require three unet evaluations. One for the first window, one for the second, and then one at an new timestep for an window that overlaps the prior two.

It is not clear to me if we need to include gradients for all three of these passes, but we will just as a first attempt. We will start with are temporal reference only model as a base, since it is already okay at blending windows.

![Shift 2 20 Steps Temporal Reference Fine-tuned No Loop](images/shift-2-20-temporal-reference-fine-tuned-no-loop-fine-tune.gif)

Nice. We can now blend windows much better in fewer time-steps.

