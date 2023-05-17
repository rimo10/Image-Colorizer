import torch
import torchvision
from torch.utils.data import DataLoader
import os
import torchvision.transforms as tt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from skimage.color import lab2rgb, rgb2lab
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2lab, lab2rgb
from torchvision import models
from PIL import Image


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 128 x 1 x 256 x 256
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 128 x 64 x 128 x 128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 128 x 128 x 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128 x 128 x 64 x 64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 128 x 256 x 64 x 64

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 128 x 256 x 32 x 32

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 128 x 512 x 32 x 32

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 128 x 512 x 32 x 32

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        # 128 x 256 x 32 x 32

    def forward(self, x):
        return self.encoder(x.float())


class Feature_Extraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inv3_img, enc_img):
        y = torch.stack([torch.stack([inv3_img], dim=2)], dim=3)
        y = y.repeat(1, 1, enc_img.shape[2], enc_img.shape[3])
        fusion_img = torch.cat((enc_img, y), axis=1)
        return fusion_img


class After_Feature_Extraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.Conv2d(1256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, fusion_img):
        return self.ft(fusion_img)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # 128 x 256 x 32 x 32

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            # 128 x 128 x 64 x 64

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 128 x 64 x 64 x 64

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            # 128 x 64 x 128 x 128

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            # 128 x 32 x 256 x 256

            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.Tanh()
            # 128 x 2 x 256 x 256
        )

    def forward(self, fuse_img):
        return self.decoder(fuse_img.float())


def rgb_to_lab(img):
    lab = rgb2lab(img.permute(0, 2, 3, 1).cpu().numpy())
    l = lab[:, :, :, 0]/50.0
    l = l.reshape(l.shape+(1,))  # Adding a new i.e color dimension
    ab = lab[:, :, :, 1:]/128
    return to_device(torch.tensor(l, dtype=torch.float).permute(0, 3, 1, 2), device), to_device(torch.tensor(ab, dtype=torch.float).permute(0, 3, 1, 2), device)


def stack_img(img):
    transform = tt.Resize(size=(299, 299))
    img = transform(img)
    img2 = img.repeat(1, 3, 1, 1)
    # img2=img2/128
    img2 = torch.tensor(img2)
    return img2


inception_v3 = to_device(models.inception_v3(pretrained=True), device)
inception_v3.aux_logits = False


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.ft_ex = Feature_Extraction()
        self.aft_ft_ex = After_Feature_Extraction()
        self.decoder = Decoder()

    def forward(self, L_img, test=False):
        inv3_img = inception_v3(stack_img(L_img))
        enc_img = self.encoder(L_img)
        fusion_img = self.ft_ex(inv3_img, enc_img)
        fusion_img = self.aft_ft_ex(fusion_img)
        dec_img = self.decoder(fusion_img)
        transform = tt.Resize(size=(256, 256))
        dec_img = transform(dec_img)
        return dec_img

    def training_step(self, batch):
        images, _ = batch
        L, AB = rgb_to_lab(images)
        out = self(L)  # 16 x 2 x 256 x 256
        loss = nn.MSELoss(reduction='mean')(out, AB)
        return loss

    def validation_step(self, batch):
        images, _ = batch
        L, AB = rgb_to_lab(images)
        out = self(L)
        loss = nn.MSELoss(reduction='mean')(out, AB)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


def get_prediction(img, color_type):
    transform = tt.Compose([
        tt.Resize((256, 256)),
        tt.ToTensor()]
    )
    img = transform(img)
    img = img.unsqueeze(0)  # 1 x 3 x 256 x 256
    model = BaseModel().to(device)
    path = str('models/'+color_type+'.pt')
    model.load_state_dict(torch.load(path, map_location='cpu'))
    L, AB = rgb_to_lab(img)
    inception_v3.eval()
    output = (model(L))
    out = torch.cat((L, output), axis=1)
    out[:, 0, :, :] = out[:, 0, :, :]*50
    out[:, 1:3, :, :] = out[:, 1:3, :, :]*128
    return lab2rgb(out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())


if __name__ == "__main__":
    img = Image.open('pictures/pexels-konstantin-mishchenko-2010812.jpg')
    pred = get_prediction(img)
    plt.imshow(pred)
    plt.show()
