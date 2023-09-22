# Longer AnimateDiff Animations Part One: Temporal Reference Only

I've had some success increasing creating longer consistent AnimateDiff animations using a trick I learned from the ControlNet "reference only" code. 

So let me catch everyone up to speed on what reference only is and how a modified version of it can help AnimateDiff stitch scenes together.

Reference only is a way to modify the self attention of a U-Net to use features from an input image. It was introduced in the ControlNet Automatic1111 extension and borrows from an inpainting trick.

So before I talk about reference only, let's quickly review how inpainting works.

### Inpainting

With inpainting you have two different types of pixels, the masked pixels you want to fill in, and the unmasked pixels you want to keep.

[![Inpainting](images/inpainting.png)]

Initially all regions start as pure noise. However, the next time-step something interesting happens. Instead of uniformly using the latents from the prior time-step, the areas that are not masked for inpainting, e.g. the areas you want to keep, those latents are overwritten.

They're overwritten with a noised version of the input image using the appropiate noise level for the current time-step. 

The end result is each time-step the latents of the unmasked area are affecting the mask area, but not vice-a-versa. The U-Net ends harmonizing the masked region to be like the input image.

The general observation is you can feed into a the U-Net latents of an input image by appropiately noising it each time-step and it will affect the other pixels that are noised normally.

### Reference Only

Reference only borrows the idea of noising up an input image but then it does something different. After it adds the appriopiate noise to the input image, it evaluates the U-Net with these noisy latents. During this evaluation it saves the features in each block of the U-Net in a buffer.

Then it does the typical U-Net evaluation of the regular noisy latents we care about for generation. Except this time in each block of the U-Net it concatenates the features from the input image it saved earlier into the self attention of the U-Net.

So the self attention now has two images features in it, the one we are generating and the input image. It causes the attention to harmonize the resultant features, essentially moving them in the direction of the input image.

### Temporal Reference Only

The motion module also uses self attention, so we can use the same trick to make the motion module use features from the prior window.

So we diffuse our first window normally, but then we save the features from the motion module. Then we diffuse the second window normally, but this time we concatenate the features from the first window into the self attention of the motion module.

We are essentially helping the motion module make the second window consistent with the first window.

### Results

As you can see it works okay by itself. There is one problem with doing this naively. If we always pass the features from the first window to the next, errors will accumulate. We can only go a single direction each time-step but by alternating the order we diffuse windows we can over the course of the diffusion process, bidirectionally inform the windows. This helps make them more consistent, and prevents auto-regressive artifacts from forming.

[![Temporal Reference Only](images/temporal-reference-only.gif)]

Still the results are not perfect. The scenes are more similar then without this trick, but there is an obvious seam. 

### The Sad State of Positional Encodings

By itself this process is helpfully but there is a problem. The motion module adds standard Transformer positional encoding to the features. So both the prior window and the current window have the same positional encoding. This is inconsistent.

Ideally we would like the prior window to have say encodings of 0-23 and the current window to have encodings of 24-47. 

Unfornately the motion module is not trainined to accomudate this. For instance if we render a 16 frame sequence but use an offset encoding, so we encode the frames as 16-31 instead of 0-15, we get this. 

![Offset Encoding](images/offset-encoding.gif)

You can see the problem. 

We can improve the situation by training, but it is ingrained into the model that it is hard to train away. Additionally, if there is a large asymetry in how it handles different positional encodings

### Modifying the Positional Encoding

We extent the motion module so it can offset the positional encoding. We want to handle to situation. Now we can train a new motion module that can incorporate the prior window features to make the current window more consistent.

### Next Steps

Right now I can make long animations that look like this, but I have to use more tricks to get there. Read part two to find out what I did.