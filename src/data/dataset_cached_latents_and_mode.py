import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from ..utils.util import zero_rank_print



class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder, mode_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            seed = 42,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.mode_folder     = mode_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        latents_path = os.path.join(self.video_folder, f"{videoid}.mp4")   
        mode_path    = os.path.join(self.mode_folder, f"{videoid}.mp4")     
        with torch.no_grad():
            pixel_values = torch.load(latents_path, map_location='cpu').detach()      
            mode_values  = torch.load(mode_path, map_location='cpu').detach()
        
        if self.is_image:
            frame_count = pixel_values.shape[0]
            # select a random frame using the generator
            frame_idx = torch.randint(frame_count, (1,), generator=self.generator).item()
            pixel_values = pixel_values[frame_idx]
        else:
            if pixel_values.shape[0] != self.sample_n_frames:
                raise "Wrong number of frames"
        
        return pixel_values, mode_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, mode_values, name = self.get_batch(idx)
                break

            except Exception as e:
                print("Exception", e)
                idx = random.randint(0, self.length-1)

        sample = dict(pixel_values=pixel_values, mode_values=mode_values, text=name)
        return sample



if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/results_2M_val.csv",
        video_folder="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/2M_val",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)