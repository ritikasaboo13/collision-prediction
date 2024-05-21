import os
import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from main import SegmentationDataSet,encoding_block,unet_model
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchmetrics

#Give Unet Saved path here
unet_model_saved_path='/scratch/sc10648/Unet/UNet/unet1.pt'
#Give validation set saved path here
val_dir='/scratch/sc10648/DL-Competition1/dataset/val/video_'

val_data_dir = [val_dir+ str(i) for i in range(1000,2000)]
val_dataset = SegmentationDataSet(val_data_dir,None)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet_model().to(DEVICE)
m = torch.load(unet_model_saved_path).state_dict()
print(m)
model.load_state_dict(m)

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(DEVICE)

y_preds_concat = None
y_trues_concat = None

y_preds_list = []
y_trues_list = []
     
def evaluate_jaccard_index(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
  
    with torch.no_grad():
        for x, y in tqdm(loader):
           
            x = x.permute(0,3,1,2).type(torch.cuda.FloatTensor).to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)

            y_preds_list.append(preds)
            y_trues_list.append(y)
            
          
            num_correct += ((preds == y) & (y != 0)).sum()
            num_pixels += (y !=0).sum()

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
           

    y_preds_concat = torch.cat(y_preds_list, dim=0)
    y_trues_concat = torch.cat(y_trues_list, dim=0)

    print(len(y_preds_list))
    print(y_preds_concat.shape)

    jac_idx = jaccard(y_trues_concat, y_preds_concat)

    print(f"Jaccard Index {jac_idx}")

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
     

val_data_subset = torch.utils.data.Subset(val_dataset, range(1000))
val_data_subset_loader = torch.utils.data.DataLoader(val_data_subset, batch_size=1, shuffle=True)

len(val_data_subset_loader)
evaluate_jaccard_index(val_data_subset_loader, model)
