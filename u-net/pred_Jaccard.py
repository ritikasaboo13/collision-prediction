import os
import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from main import SegmentationDataSet,encoding_block,unet_model
#from main_unet3 import SegData,EncodingBlock,unet_model
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchmetrics
from torchvision import transforms
import re

#Give Unet Saved path here
unet_model_saved_path='/scratch/ak11089/final-project/Unet/unet-test/unet1.pt'
#Give validation set saved path here
# val_dir='/scratch/sc10648/DL-Competition1/dataset/val/video_'

# val_data_dir = [val_dir+ str(i) for i in range(1000,2000)]
# val_dataset = SegmentationDataSet(val_data_dir,None)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#model = unet_model().to(DEVICE)
#m = torch.load(unet_model_saved_path).state_dict()
#model.load_state_dict(m)
#model.eval()

class PredictedImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        #print(self.images[idx])
        pattern = re.compile(r'pred_(\d+)$')
        match = pattern.search(self.images[idx])
        #print(self.images[idx][-8:-4])
        video_number = int(self.images[idx][-8:-4])
        #print(video_number)
        
        if self.transform:
            image = self.transform(image)
        
        return image, video_number

class UnNormalize(object):
    def __call__(self, tensor):
        return tensor * 255

# Load Predicted Images Dataset
#pred_dataset = PredictedImageDataset(directory='/scratch/sc10648/Unet/diffusion/pred_val', transform = transforms.Compose([transforms.Resize((160,240)), transforms.ToTensor(),UnNormalize()]))
#pred_dataset = PredictedImageDataset(directory='/scratch/ak11089/final-project//Deep-Learning-Project-Fall-23/src/mcvd/prev_val5v6/', transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor(),UnNormalize()]))
#pred_dataset = PredictedImageDataset(directory='/scratch/ak11089/final-project//hidden-out/11v1-cont-1000-1050-val-pred/', transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor(),UnNormalize()]))

#pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

# def load_validation_masks(val_dir, start_index, end_index):
#     validation_masks = []
#     for i in range(start_index, end_index):
#         mask_path = f'{val_dir}{i}/mask.npy'
#         masks = np.load(mask_path)
#         validation_masks.append(masks[21])  # Load the 22nd mask
#     return validation_masks

# val_masks = load_validation_masks('/scratch/sc10648/DL-Competition1/dataset/val/video_', 1000, 2000)

def load_validation_masks(val_dir, start_index, end_index, new_size=(128, 128)):
    validation_masks = []
    resize_transform = transforms.Resize(new_size)
    for i in range(start_index, end_index):
        mask_path = f'{val_dir}{i}/mask.npy'
        mask = np.load(mask_path)
        mask_image = Image.fromarray(mask[21])  # Convert the 22nd mask to an image
        #mask_resized = resize_transform(mask_image)  # Resize the mask
        validation_masks.append(np.array(mask_image))
    return validation_masks

#val_masks = load_validation_masks('/scratch/ak11089/final-project//raw-data-1/dataset/val/video_', 1000, 2000, new_size=(128, 128))


jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

# softmax = torch.nn.Softmax(dim=1)
jaccard_scores = []
total_dp = 0

def validate(val_path, ckpt):
    result = predict(ckpt, val_path)
    # Convert numpy array to torch tensor for validation mask
    val_masks = load_validation_masks('/scratch/ak11089/final-project//raw-data-1/dataset/val/video_', 1000, 2000, new_size=(128, 128))
    preds = []
    masks = []
    tt = 0
    avg_jac = 0.0
    for vn in result:
            
        val_mask_tensor = torch.from_numpy(val_masks[vn-1000]).unsqueeze(0)
        masks.append(val_mask_tensor)
        preds.append(result[vn].to("cpu"))
        avg_jac += jaccard(result[vn].to("cpu"), val_mask_tensor)
        #print(val_mask_tensor.shape, result[vn].shape)
        tt += 1
    # Compute Jaccard Index for the current pair of masks
    #print(preds)
    print("Sujana's Jaccard", avg_jac /tt)
    preds = torch.concat(preds, dim = 0)
    masks = torch.concat(masks, dim = 0)
    print("final pred shape",preds.shape)
    score = jaccard(preds, masks)
    print("jacc score", score)
    #jaccard_scores.append(score.item())
    #total_dp += 1
    #if total_dp > 20:
        #break

def predict(ckpt, data_path):
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = unet_model().to(DEVICE)
    m = torch.load(ckpt).state_dict()
    model.load_state_dict(m)
    model.eval()
    pred_dataset = PredictedImageDataset(directory=data_path, transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor(),UnNormalize()]))
    #pred_dataset = PredictedImageDataset(directory=data_path, transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor()]))
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
    results = {}
    for i, (image, video_number) in enumerate(tqdm(pred_loader, desc="Generating Masks")):
        image = image.to(DEVICE)
        with torch.no_grad():
            output = model(image)
            preds = torch.argmax(output, axis=1)
        print(preds.shape)
        for j in range(video_number.shape[0]):
            results[video_number[j].item()] = preds[j].unsqueeze(0)



    return results

def save_results_pt(result):
    video_nums = list(result.keys())
    video_nums.sort()
    start = 15000
    idx = 0
    res_list = []
    for vn in video_nums:
        print(vn)
        assert vn == start, f"{start} is not present in your prediction"
        pred = result[vn]
        res_list.append(pred.to("cpu"))
        start += 1

    print("total dp", start)
    assert start == 17000
    result_tensor = torch.concat(res_list, dim = 0)
    print("tensor final shape",result_tensor.shape)
    torch.save(result_tensor, "final_leaderboard_team_27.pt")


#r = predict(unet_model_saved_path,"/scratch/ak11089/final-project//final_pred_hidden/all/" )
#save_results_pt(r)

#validate("/scratch/ak11089/final-project//hidden-out/11v1-cont-1000-1050-val-pred/", unet_model_saved_path)
validate("/scratch/ak11089/final-project/val_ad", unet_model_saved_path)
#average_jaccard_index = sum(jaccard_scores) / len(jaccard_scores)
#print(f'Average Jaccard Index: {average_jaccard_index}')
#print(f"total images tested on {total_dp}")

