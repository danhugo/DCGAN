import sys
import os

sys.path.append(os.path.abspath('.'))

import time
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import src.utils.visualization as visual
from src import BASE_DIR
from src.utils.device import get_device
from src.data.LSUN.dataset import LSUNDataset
import src.data.LSUN.transforms as Ltransforms
from src.models import LSUN_MODEL_DIR, LSUN_REPORT_DIR, LSUN_REPORT_IMAGE_DIR, CONFIG_PATH
import src.utils.wb as wb
import src.utils.config as cf
from src.utils.instance import dict_to_object

IMAGES_PER_ROW = 10
NOISE_DIM = 100

class Generator(nn.Module):
    def __init__(self,num_channel: int = 3, ngf: float = 128) -> None:
        """# DCGAN Generator
    
        Arguments
        ---------
        - num_channel (int): number of channel or dimension of input (default: 3)
        - ngf (int): number of generation filters (default: 128)
        
        Input
        -----
        - input (tensor): size N x 100 with N is number of points which are 
            sampled from 100-dimension distribution.

        Returns
        -------
        - output (tensor): size N x 3 x 64 x 64, which are N images with 
            size 3 x 64 x 64.

        """
        nz = 100 # num of noise dim
        ngf = 128 # num of gen filter
        super(Generator,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4 , 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.deconv5 = nn.ConvTranspose2d(ngf, num_channel, 4, 2, 1, bias=False)
        
        # self.apply(self._init_weights)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean = 0, std = 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x[:,:,None,None]
        x = F.relu(self.bn1(self.deconv1(x))) # 1 x 1 -> 4 x 4
        x = F.relu(self.bn2(self.deconv2(x))) # 4 x 4 -> 8 x 8
        x = F.relu(self.bn3(self.deconv3(x))) # 8 x 8 -> 16 x 16
        x = F.relu(self.bn4(self.deconv4(x))) # 16 x 16 -> 32 x 32
        x = torch.tanh(self.deconv5(x))       # 32 x 32 -> 64 x 64
        
        return x

class Discriminator(nn.Module):
    def __init__(self, num_channel: int = 3, ndf: float = 128) -> None:
        """# Discriminator

        Arguments
        ---------
        - num_channel (int): number of channel or dimension of input (default: 3)
        - ndf (float): number of discriminator filters (default: 128)

        Inputs
        ------
        - input (tensor): N x 3 x 64 x 64, N images with size 3 x 64 x 64

        Returns
        - output (tensor): N, vector N discriminative results of N images 
        -------
        
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        # self.apply(self._init_weights)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean = 0, std = 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope = 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope = 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope = 0.2)
        x = torch.sigmoid(self.conv5(x))
        x = x.squeeze()
        
        return x

def config_parser():
    config = cf.read_config(CONFIG_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default=config.project,
                        help='name of project')
    parser.add_argument('--train_size', type=int, default=config.train_size,
                        help='training data size')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                        help='batch size')
    parser.add_argument('--noise_dim', type=int, default=config.noise_dim,
                        help='number dimensions of noise')
    parser.add_argument('--num_channel', type=int, default=config.num_channel,
                        help='number channels of the image')
    parser.add_argument('--ngf', type=int, default=config.ngf,
                        help='number of generation filter')
    parser.add_argument('--ndf', type=int, default=config.ndf,
                        help='number discriminator filter')
    parser.add_argument('--num_epoch', type=int, default=config.num_epoch,
                        help='number of epoches for training')
    parser.add_argument('--lr', type=float, default=config.lr,
                        help="learning rate")
    parser.add_argument('--beta1', type=float, default=config.beta.beta1,
                        help='Adam optimizer beta 1')
    parser.add_argument('--beta2', type=float, default=config.beta.beta2,
                        help='Adam optimizer beta 2')
    parser.add_argument('--rows_gen_image', type=int, default=config.rows_gen_image,
                        help='number rows for a grid of generated images, each contains 10 images')
    return parser

def train():
    """Train model."""
    # get arguments
    parser = config_parser()
    args = parser.parse_args()
    args.experiment_name = f's{args.train_size}-bs{args.batch_size}-ep{args.num_epoch}'
    endlog = {
        "Gloss" : 0.0,
        "Dloss" : 0.0,
        "lr" : args.lr,
        "epoch": 0
    }
    # wandb init
    wb.on_pretrain_start(args)

    # load data
    dataloader = load_data(size=args.train_size, batch_size=args.batch_size)
    print('Finish loading data')

    device = get_device()
    print(f"You are using {device}")

    # declare model
    G = Generator(args.num_channel).to(device)
    D = Discriminator(args.num_channel).to(device)
    G._init_weights()

    optimG = optim.Adam(G.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    optimD = optim.Adam(D.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    criterion = nn.BCELoss()

    z_dim = NOISE_DIM

    # fixed noise for generated image after each epoch
    # to compare what model has learned
    fixed_noise = torch.randn((args.rows_gen_image * IMAGES_PER_ROW, z_dim)).to(device)
    grid_images = []
    global_step = 0
    for epoch in range(args.num_epoch):
        err_Gs = []
        err_Ds = []
        epoch_start_time = time.time()
        
        for step, data in enumerate(dataloader):
            batch_size = data.size(0)
            noise = torch.randn((batch_size,z_dim)).to(device)
            data = data.to(device)
            # optimize D
            # optimize with true
            D.zero_grad()
            output = D(data)
            labels = torch.ones((batch_size,)).to(device)
            err_D_true = criterion(output, labels)

            # optimize with fake
            fake_samples = G(noise)
            # nothing to do with G or its output when optimize D
            # then detach it
            output = D(fake_samples.detach()) 
            labels = torch.zeros((batch_size,)).to(device)
            err_D_fake = criterion(output, labels)

            err_D = err_D_true + err_D_fake
            err_Ds.append(err_D)
            err_D.backward()
            optimD.step()

            # optimize G
            G.zero_grad()
            output = D(fake_samples)
            labels = torch.ones((batch_size,)).to(device)
            err_G = criterion(output, labels)
            err_Gs.append(err_G)
            err_G.backward()
            optimG.step()
            global_step += 1
            wb.on_training_epoch(dict_to_object({
                'Gloss': err_G,
                'Dloss': err_D,
                'D_real_loss': err_D_true,
                'D_fake_loss': err_D_fake,
                'epoch': (step + 1 + (len(dataloader) * epoch))/len(dataloader),
                'step': global_step,
                }))
        
        per_epoch_time = time.time() - epoch_start_time
        mean_G_loss, mean_D_loss = torch.mean(torch.FloatTensor(err_Gs)), torch.mean(torch.FloatTensor(err_Ds))
        print(f"[{epoch+1}/{args.num_epoch}: {per_epoch_time // 60:.0f}m{per_epoch_time % 60:.0f}s] \
              errG: {mean_G_loss.item():.6f}, errD:{mean_D_loss.item():.6f}")
        
        # check pointing for every epoch
        torch.save(G.state_dict(), os.path.join(LSUN_MODEL_DIR, f'gen_params_{epoch+1}.pth'))
        torch.save(G.state_dict(), os.path.join(LSUN_MODEL_DIR, f'dis_params_{epoch+1}.pth'))

        # save the image ouput every epoch
        fake_samples = G(fixed_noise)
        grid_image = visual.create_grid_image(fake_samples, save = True, path_to_save=os.path.join(LSUN_REPORT_IMAGE_DIR, f'LSUN_epoch_{epoch+1}.png'))
        grid_images.append(grid_image)
        
        endlog['step'] = global_step
        endlog['Gloss'], endlog['Dloss'] = mean_G_loss, mean_D_loss
        endlog['gen_image'] = {'name': f'LSUN_epoch_{epoch+1}', 'data': grid_image}
        wb.on_train_epoch_end(dict_to_object(endlog))

    visual.save_gif(os.path.join(LSUN_REPORT_DIR ,f'LSUN_{args.num_epoch}_epoch.gif'),grid_images)

def load_data(size: int = 100000, batch_size: int = 128) -> LSUNDataset:
    """Return dataloder of LSUN dataset with a chosen size
    
    Attributes
    - size (int): size of dataset to be load (default: 100000)
    - batch_size (int): batch size (default: 128)
    Return
    ------
    (DataLoader): dataloader of Lsun dataset which comprises list of data (not (data, label))
    """
    composed = transforms.Compose([
        Ltransforms.CenterCrop(),
        Ltransforms.ReScale(64),
        transforms.ToTensor()
    ])
    
    dataset = LSUNDataset(os.path.join(BASE_DIR, 'data/LSUN/bedroom_train_data'), nsample=size, transform=composed)
    train_loader = DataLoader(dataset, batch_size= batch_size, shuffle= True)
    return train_loader

if __name__ == '__main__':
    train()
