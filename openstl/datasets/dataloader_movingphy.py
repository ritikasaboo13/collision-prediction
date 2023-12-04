from PIL import Image
from IPython.display import display
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

import gzip
import numpy as np
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader


class MovingPhysics(Dataset):
    def __init__(self, root, is_train=True, data_name='moving_physics', pre_frame_length=11, aft_frame_length=11, image_height=160, image_width=240, 
                 transform=None, use_augment=False, normalize=False):
        super(MovingPhysics, self).__init__()
        self.dataset = None
        self.is_train = is_train
        self.data_name = data_name
        self.image_height = image_height
        self.image_width = image_width
        self.pre_frame_length = pre_frame_length
        self.aft_frame_length = aft_frame_length
        self.total_frames = self.pre_frame_length + self.aft_frame_length
        self.transform = transform
        self.train_fdr_path = None
        self.unlabeled_fdr_path = None
        self.val_fdr_path = None
        self.test_fdr_path = None
        self.videos = []
        if self.is_train:
          self.train_fdr_path = os.path.join(root, "train")
          self.unlabeled_fdr_path = os.path.join(root, "unlabeled")
         # print("Train folder path: ", self.train_fdr_path)
         # print("Unlabeled folder path: ", self.unlabeled_fdr_path)
          train_videos = os.listdir(self.train_fdr_path)
          unlabeled_videos = os.listdir(self.unlabeled_fdr_path)
          for v in train_videos:
              self.videos.append(os.path.join(self.train_fdr_path, v))
          for v in unlabeled_videos:
            self.videos.append(os.path.join(self.unlabeled_fdr_path, v))

         # print("Train video paths:", self.train_videos)
         #  print("Unlabeled video paths:", self.unlabeled_videos)
         #  self.videos = self.train_videos + self.unlabeled_videos
        else:
          self.val_fdr_path = os.path.join(root, "val")
          #print("Val folder path: ", self.val_fdr_path)
          #self.test_fdr_path = self.val_fdr_path
          val_videos = os.listdir(self.val_fdr_path)
          for v in val_videos:
            self.videos.append(os.path.join(self.val_fdr_path, v))
          #print("Val video paths:", self.val_videos
                
                    
          #self.videos = self.val_videos

        self.videos.sort() # do I need to sort it? 
       # print(self.videos[:11])
        self.mean = 0
        self.std = 1


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
      video = self.videos[index]
      #print(video)
      frames = []

      for f in range(self.total_frames):
        frame = Image.open(os.path.join(video, "image_"+str(f)+".png"))
        transform = transforms.ToTensor()
        frame = transform(frame)
       # print(frame)
        frame = frame/255
        frames.append(frame)
        #print(frame)

      frames_ = torch.stack(frames)
      #print(frames_.shape)
     # print(frames_[:self.pre_frame_length, ...].shape)
      #print(frames_[self.pre_frame_length:, ...].shape)
      return frames_[:self.pre_frame_length, ...], frames_[self.pre_frame_length:, ...]

def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='mnist',
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    image_size = in_shape[-1] if in_shape is not None else 64
    train_set = MovingPhysics(root=data_root, is_train=True, data_name=data_name, 
                              pre_frame_length=pre_seq_length, aft_frame_length=aft_seq_length, image_height=in_shape[-2], 
                              image_width=in_shape[-1], transform=None, use_augment=False, normalize=True)
    val_set = MovingPhysics(root=data_root, is_train=False, data_name=data_name, 
                              pre_frame_length=pre_seq_length, aft_frame_length=aft_seq_length, image_height=in_shape[-2], 
                              image_width=in_shape[-1], transform=None, use_augment=False, normalize=True)

    dataloader_train = create_loader(train_set,
                                    batch_size=batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed
                                     )
    dataloader_vali = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed
                                    )
    dataloader_test = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed
                                    )
    
   # print("Dataloader shapes: ", dataloader_train.shape, dataloader_vali.shape, dataloader_test.shape)

    return dataloader_train, dataloader_vali, None
