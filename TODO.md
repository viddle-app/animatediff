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
