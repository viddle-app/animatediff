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

Need to write a new captioning dataset script
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