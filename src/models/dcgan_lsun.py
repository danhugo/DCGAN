import sys
import os
import datetime
sys.path.append(os.path.abspath('.'))

import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.utils.visualization as visual
from src import BASE_DIR
from src.utils.device import get_device
from src.data.LSUN.dataset import LSUNDataset
import src.data.LSUN.transforms as Ltransforms
from src.models import LSUN_MODEL_DIR, LSUN_REPORT_DIR, LSUN_REPORT_IMAGE_DIR, CONFIG_PATH
import src.utils.wb as wb
import src.utils.config as cf

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
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4 , 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(ngf, num_channel, 4, 2, 1, bias=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean = 0, std = 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x[:,:,None,None]
        x = self.relu1(self.bn1(self.deconv1(x))) # 1 x 1 -> 4 x 4
        x = self.relu2(self.bn2(self.deconv2(x))) # 4 x 4 -> 8 x 8
        x = self.relu3(self.bn3(self.deconv3(x))) # 8 x 8 -> 16 x 16
        x = self.relu4(self.bn4(self.deconv4(x))) # 16 x 16 -> 32 x 32
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
        - input (tensor): N x C x H x W, N images with size C x H x W

        Returns
        - output (tensor): N, vector N discriminative results of N images 
        -------
        
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, ndf, 4, 2, 1, bias=True)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.leaky_relu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean = 0, std = 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.leaky_relu3(self.bn3(self.conv3(x)))
        x = self.leaky_relu4(self.bn4(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        x = x.squeeze()
        
        return x

def train(args):
    """Train model."""
    # log parameter
    end_epoch_log = {
        "end/Gloss" : 0.0,
        "end/Dloss" : 0.0,
        "end/lr" : args.lr,
    }

    in_epoch_log = {
        "in/D_real_loss": 0.0,
        "in/D_fake_loss": 0.0,
        "in/D_loss": 0.0,
        "in/G_loss": 0.0,
    }

    # wandb init
    wb.on_train_start(args)

    # load data
    dataloader = load_data(size=args.train_size, batch_size=args.batch_size)
    print('Finish loading data loader')
    
    # get device using for training
    device = get_device()
    print(f"device: {device}")

    # declare model
    # torch.manual_seed(26)
    G = Generator(args.num_channel, args.ngf).to(device)
    D = Discriminator(args.num_channel, args.ndf).to(device)
    G._init_weights()
    z_dim = args.noise_dim

    optimG = optim.Adam(G.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    optimD = optim.Adam(D.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    
    # loss fnc
    criterion = nn.BCELoss()

    # fixed noise for generated image after each epoch
    # to compare what model has learned
    fixed_noise = torch.randn((args.rows_gen_image * args.images_per_row, z_dim)).to(device)
    grid_images = []
    global_step = 0

    for epoch in range(args.num_epoch):
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        
        for data in tqdm(dataloader, desc=f'epoch {epoch + 1}'):
            batch_size = data.size(0)
            noise = torch.randn((batch_size,z_dim)).to(device)
            data = data.to(device)
            # optimize D
            # optimize with true
            D.zero_grad(set_to_none=True)
            output = D(data)
            labels = torch.ones((batch_size,)).to(device)
            D_real_loss = criterion(output, labels)

            # optimize with fake
            fake_samples = G(noise)
            # nothing to do with G or its output when optimize D
            # then detach it
            output = D(fake_samples.detach()) 
            labels = torch.zeros((batch_size,)).to(device)
            D_fake_loss = criterion(output, labels)

            D_loss = D_real_loss + D_fake_loss
            D_losses.append(D_loss)
            D_loss.backward()
            optimD.step()

            # optimize G
            G.zero_grad(set_to_none=True)
            output = D(fake_samples)
            labels = torch.ones((batch_size,)).to(device)
            G_loss = criterion(output, labels)
            G_losses.append(G_loss)
            G_loss.backward()
            optimG.step()

            # log in epoch
            global_step += 1
            if global_step % 100 == 0:
                in_epoch_log['in/D_fake_loss'] = D_fake_loss
                in_epoch_log['in/D_real_loss'] = D_real_loss
                in_epoch_log['in/D_loss'] = D_loss
                in_epoch_log['in/G_loss'] = G_loss
                wb.in_train_epoch(in_epoch_log, global_step)
        
        per_epoch_time = time.time() - epoch_start_time
        G_mean_loss, D_mean_loss = torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(D_losses))
        print(f"[{epoch+1}/{args.num_epoch}: {per_epoch_time // 60:.0f}m{per_epoch_time % 60:.0f}s] \
              errG: {G_mean_loss.item():.6f}, errD:{D_mean_loss.item():.6f}")
        
        # check pointing for every epoch
        torch.save(G.state_dict(), os.path.join(SAVE_MODEL_DIR, f'gen_params_{epoch+1}.pth'))
        torch.save(D.state_dict(), os.path.join(SAVE_MODEL_DIR, f'dis_params_{epoch+1}.pth'))

        # save the image ouput every epoch
        with torch.no_grad():
            fake_samples = G(fixed_noise)
            image_name = f'LSUN_epoch_{epoch+1}'
            grid_image = visual.create_grid_image(fake_samples, nimage_row=args.images_per_row, save = True, path_to_save=os.path.join(LSUN_REPORT_IMAGE_DIR, f'{image_name}.png'))
            grid_images.append(grid_image)
        
        # log end epoch
        end_epoch_log['end/Gloss'], end_epoch_log['end/Dloss'] = G_mean_loss, D_mean_loss
        wb.end_train_epoch(end_epoch_log, global_step)
        wb.log_image({image_name: grid_image})

    visual.save_gif(os.path.join(LSUN_REPORT_DIR ,f'LSUN_{args.num_epoch}_epoch.gif'), grid_images)

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
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle= True, num_workers=1, pin_memory=torch.cuda.is_available())
    return train_loader

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
                        help='number rows for a grid of generated images, default each contains 10 images')
    parser.add_argument('--images_per_row', type=int, default=config.images_per_row,
                        help='number of images per row')
    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.experiment_name = f's{args.train_size}_bs{args.batch_size}_ep{args.num_epoch}'
    now = datetime.datetime.now()
    SAVE_MODEL_DIR = os.path.join(BASE_DIR,'models/LSUN', now.strftime("%Y%m%d_%H%M_") + args.experiment_name)
    os.mkdir(SAVE_MODEL_DIR)
    train(args)
