TODO now
Frame Interpolator
- prepare the data set for training the 8 channel unet
- get the diffusers training code

Try training without the 1/sqrt(snr)
Try training with a unet that has been fine-tuned for 1/sqrt(snr)
Compare miniSD and non-miniSD fine-tuned with data set and not
Fine-tune the dreamshaper with 1/sqrt(snr) for use after
Contain training from scratch with old loss 4
refactor code to make it easier to switch out losses for validation
use more videos for validation
prepare the trailer dataset

retrain with 1e-9 and a proper schedule
try using an array of lrs for different timestep ranges to fine-tune just the high end
try use noise perputations

log out latents when rendering for debugging


OLD:
- Clean up the partition code
- Add to controlnet flow
- Enable img2img flow
- Setup consistent Euler sampling
- Add lora loading
- Use image init (try it on frame and back frames for looping)
- Try multidiffusion
- Fix way it makes wrap around so it is more fair. Make windows and increment them instead.
- the window size is causing distinct sections. This is probably because
  early on it barely moves which allows sections to form. Either use multidiffusion or a 
  bigger increments at least early on.

Ideas
Looping to the music is cool
Looping and reactive would be interesting
If the fidelity could be increased you could animate a whole alblum. 

Try multidiffusion with less blending
- doesn't seem like it works at least with noise_pred blending
Modify to use same size windows without looping and use blending instead
- This is going to have issues
Look at how the inpainting model is trained


Should start downloading videos all the time
Docker image for training

Finish second blog post
Add images

First port over cross attention

We could look at collaborative score distillation
As a way to using motion astetic classier

to guide the sample process

We need to train a aestehtic model for video clips
How can we train that?

We need a data set that scores video clips by voting
The goal is build a video voting system
that makes data that is used to train a classifier

we can acheive blending by sometimes
have the window be different lengths

don't shift a window
shift how much they overlap. 

The model for image 
We need to train a model that get the extra channels and tries to predict the sequence of frames.
It should get some subset of frames and predict the rest.

The spectrogram should remain the same after fine-tuning
the average should stay the same
and the variance

because weight not
I should have added pe to the input frames
I need a way to add a list of pe
I should only every have a 16 by itself if it is tzero. 

I should train in a range otherwise. Because it can't really resolve really deal with a difference. 
I might need to evaluate to that point. 
It needs to learn how to handle repeated diffusion steps or is it idepotent


