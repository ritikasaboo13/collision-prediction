import os
import torch
import torch.nn as nn
import numpy as np
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
from torch.optim import lr_scheduler


class SegData(Dataset):

    def __init__(self, videos, transform=None):
        self.transforms = transform
        self.images, self.masks = [], []
        for i in videos:
            imgs = os.listdir(i)
            self.images.extend([i + '/' + img for img in imgs if not img.startswith('mask')]) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]))/255
        x = self.images[idx].split('/')
        image_name = x[-1]
        mask_idx = int(image_name.split("_")[1].split(".")[0])
        x = x[:-1]
        mask_path = '/'.join(x)
        mask = np.load(mask_path + '/mask.npy')
        mask = mask[mask_idx, :, :]

        if self.transforms is not None:
            mod = self.transforms(image=img, mask=mask)
            img = mod['image']
            mask = mod['mask']

        return img, mask


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        m = []
        m.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        m.append(nn.BatchNorm2d(out_channels))
        m.append(nn.ReLU(inplace=True))
        m.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        m.append(nn.BatchNorm2d(out_channels))
        m.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*m)

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(self, out_channels=49, channels=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = EncodingBlock(3, channels[0])
        self.conv2 = EncodingBlock(channels[0], channels[1])
        self.conv3 = EncodingBlock(channels[1], channels[2])
        self.conv4 = EncodingBlock(channels[2], channels[3])
        self.conv5 = EncodingBlock(channels[3] * 2, channels[3])
        self.conv6 = EncodingBlock(channels[3], channels[2])
        self.conv7 = EncodingBlock(channels[2], channels[1])
        self.conv8 = EncodingBlock(channels[1], channels[0])
        self.upconv1 = nn.ConvTranspose2d(channels[3] * 2, channels[3], kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.bottleneck = EncodingBlock(channels[3], channels[3] * 2)
        self.final_layer = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # print("Initial Input:", x.shape)
        
        x = self.conv1(x)
        # print("After conv1:", x.shape)
        skip_connections.append(x)
        x = self.pool(x)
        # print("After pool1:", x.shape)
        
        x = self.conv2(x)
        # print("After conv2:", x.shape)
        skip_connections.append(x)
        x = self.pool(x)
        # print("After pool2:", x.shape)
        
        x = self.conv3(x)
        # print("After conv3:", x.shape)
        skip_connections.append(x)
        x = self.pool(x)
        # print("After pool3:", x.shape)
        
        x = self.conv4(x)
        # print("After conv4:", x.shape)
        skip_connections.append(x)
        x = self.pool(x)
        # print("After pool4:", x.shape)
        
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        x = self.upconv1(x)
        # print("After upconv1:", x.shape)
        x = torch.cat((skip_connections[0], x), dim=1)
        # print("After concat with skip_connections[0]:", x.shape)
        x = self.conv5(x)
        # print("After conv5:", x.shape)
        
        x = self.upconv2(x)
        # print("After upconv2:", x.shape)
        x = torch.cat((skip_connections[1], x), dim=1)
        # print("After concat with skip_connections[1]:", x.shape)
        x = self.conv6(x)
        # print("After conv6:", x.shape)
        
        x = self.upconv3(x)
        # print("After upconv3:", x.shape)
        x = torch.cat((skip_connections[2], x), dim=1)
        # print("After concat with skip_connections[2]:", x.shape)
        x = self.conv7(x)
        # print("After conv7:", x.shape)
        
        x = self.upconv4(x)
        # print("After upconv4:", x.shape)
        x = torch.cat((skip_connections[3], x), dim=1)
        # print("After concat with skip_connections[3]:", x.shape)
        x = self.conv8(x)
        # print("After conv8:", x.shape)
        
        x = self.final_layer(x)
        # print("After final_layer:", x.shape)
        
        return x

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(DEVICE)
    y_preds_list = []
    y_trues_list = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)

            y_preds_list.append(preds)
            y_trues_list.append(y)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

          
    y_preds_concat = torch.cat(y_preds_list, dim=0)
    y_trues_concat = torch.cat(y_trues_list, dim=0)
    print("IoU over val: ", mean_iou)

    print(len(y_preds_list))
    print(y_preds_concat.shape)

    jac_idx = jaccard(y_trues_concat, y_preds_concat)

    print(f"Jaccard Index {jac_idx}")

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")


def calculate_batch_iou(SMOOTH, outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2)) 
    union = (outputs | labels).float().sum((1, 2)) 

    iou = (intersection + SMOOTH) / (union + SMOOTH) 

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  

    return thresholded 


if __name__ == "__main__":

    train_data_path = '/scratch/sc10648/DL-Competition1/dataset/train/video_' 
    val_data_path = '/scratch/sc10648/DL-Competition1/dataset/val/video_'

    train_data_dir = [train_data_path + str(i) for i in range(0, 1000)]
    train_data = SegData(train_data_dir, None)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

    val_data_dir = [val_data_path + str(i) for i in range(1000, 2000)]
    val_data = SegData(val_data_dir, None)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = unet_model().to(DEVICE)
    
    best_model = None
    
    epochs = 40
    lr = 1e-4
    SMOOTH = 1e-6
    
    epochs_no_improve = 0
    stopping = False
    patience = 3

    floss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scaler = torch.cuda.amp.GradScaler()

    # Train loop
    for epoch in range(epochs):
        loop = tqdm(train_dataloader)
        for idx, (data, targets) in enumerate(loop):
            data = data.permute(0, 3, 1, 2).to(torch.float16).to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = floss(predictions, targets)

           
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        scheduler.step() 
        losses = []
        latest_loss = 1000000
        model.eval()
        mean_iou = []
        iou_list = []
        last_iou = 0
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_dataloader)):
                x = x.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor).to(DEVICE)
                y = y.to(DEVICE)
                y = y.type(torch.long)
               
                with torch.cuda.amp.autocast():
                    preds = model(data)
                    loss_v = floss(preds, y)

                losses.append(loss_v.item())
                preds_arg = torch.argmax(softmax(preds), axis=1)

                thresholded_iou = calculate_batch_iou(SMOOTH, preds_arg, y)
                iou_list.append(thresholded_iou)

            mean_iou = sum(iou_list) / len(iou_list)
            avarage_loss = sum(losses) / len(losses)
            
            print(f"Epoch: {epoch}, avgerage IoU: {mean_iou}, average loss: {avarage_loss}")

        if avarage_loss < latest_loss:
            best_model = model
            torch.save(best_model, 'unet5.pt')
            latest_loss = avarage_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience and epoch > 10:
            stopping = True
            print("Exiting")

    check_accuracy(val_dataloader, best_model)

