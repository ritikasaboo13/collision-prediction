import os
import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from main import SegmentationDataSet,encoding_block,unet_model
import torch
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchmetrics

# Load the predicted images
pred_images = np.load('/path/to/prediction_file.npy')
pred_images_tensor = torch.tensor(pred_images, dtype=torch.float32)

# Create a dataset and dataloader for the predicted images
pred_dataset = TensorDataset(pred_images_tensor)
pred_loader = DataLoader(pred_dataset, batch_size=1)

#Give Unet Saved path here
unet_model_saved_path='/scratch/sc10648/Unet/UNet/unet1.pt'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet_model().to(DEVICE)
m = torch.load(unet_model_saved_path).state_dict()
model.load_state_dict(m)
model.eval()

softmax = torch.nn.Softmax(dim=1)
generated_masks = []

for images in pred_loader:
    images = images[0].to(DEVICE)  
    with torch.no_grad():
        output = model(images)
        preds = torch.argmax(softmax(output), dim=1)
        generated_masks.append(preds.numpy())


def load_validation_masks(val_dir, start_index, end_index):
    validation_masks = []
    for i in range(start_index, end_index):
        mask_path = f'{val_dir}{i}/mask.npy'
        masks = np.load(mask_path)
        validation_masks.append(masks[21])  # Load the 22nd mask, assuming zero-based indexing
    return validation_masks

val_masks = load_validation_masks('//scratch/sc10648/DL-Competition1/dataset/val/video_', 1000, 2000)

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(DEVICE)

jaccard_scores = []

for i, gen_mask in enumerate(generated_masks):
    # Convert numpy array to torch tensor for validation mask
    val_mask = torch.from_numpy(val_masks[i]).unsqueeze(0).to(DEVICE)
    gen_mask_tensor = torch.from_numpy(gen_mask).unsqueeze(0).to(DEVICE)

    # Compute Jaccard Index for the current pair of masks
    score = jaccard(gen_mask_tensor, val_mask)
    jaccard_scores.append(score.item())

# Calculate average Jaccard Index
average_jaccard_index = sum(jaccard_scores) / len(jaccard_scores)
print(f'Average Jaccard Index: {average_jaccard_index}')
