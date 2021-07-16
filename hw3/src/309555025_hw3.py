#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # use to show images
import matplotlib.image as mpimg # use to read images
import cv2
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# In[2]:


#
# Generator & Discriminator model
#
class Discriminator(nn.Module):
    def __init__(self,channels_img):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, 128, kernel_size=4, stride=2, padding=1 # 3*64*64 => 128*32*32
            ),
            nn.LeakyReLU(0.2),
            self.block(128, 256, 4, 2, 1),                 # 128*32*32 => 256*16*16
            self.block(256, 512, 4, 2, 1),                 # 256*16*16 => 512*8*8
            self.block(512, 1024, 4, 2, 1),                # 512*8*8 => 1024*4*4
            nn.Conv2d(1024, 1, 4, 2, 0),                   # 4*4*1024 => 1*1*1
            nn.Sigmoid(),                                  # converting value to [0,1]
        )
        self.initialize_weights()
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    def forward(self,x):
        return self.disc(x)
class Generator(nn.Module):
    def __init__(self,z_dim, channels_img):
        super(Generator,self).__init__()
        # Input: N x 100 x 1 x 1
        self.gen = nn.Sequential(
            self.block(z_dim, 1024, 4, 1, 0),       # 100*1*1 => 1024*4*4
            self.block(1024, 512, 4, 2, 1),         # 1024*4*4 => 512*8*8
            self.block(512, 256, 4, 2, 1),          # 512*8*8 => 256*16*16
            self.block(256, 128, 4, 2, 1),          # 256*16*16 => 128*32*32
            nn.ConvTranspose2d(                     # 128*32*32 => 3*64*64
                128, 
                channels_img, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),      
            nn.Tanh(),
        )
        self.initialize_weights()
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    def forward(self,x):
        return self.gen(x)


# In[3]:


# Hyperparameter
LR = 0.0002
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100

# Data preprocessing
transforms_ = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5]),
    ]
)


# In[4]:


# Read images
img = Image.open("./crops/image_0001599_crop_0000008.png").convert('RGB')
img1 = transforms_(img)
print(img1.shape)
file = open('file_name.txt','r')
file_name = []
while(True):
    line = file.readline()
    if(not line):
        break
    file_name.append(line.split('\n')[0])
file.close()
real_images= []
for name in file_name:
    img = Image.open('./crops/'+name).convert('RGB')
    real_images.append(transforms_(img))


# In[5]:


# Dataloader, model, Optimizer
dataloader = DataLoader(real_images, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG).to(device)
disc = Discriminator(CHANNELS_IMG).to(device)

opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5,0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5,0.999))
loss_func = nn.BCELoss()

# make fixed noise
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# To show result
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

# set training mode
gen.train()
disc.train()
step = 0

# training process
for epoch in range(NUM_EPOCHS):
    for i, real in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise).to(device)
        
        # Train Discriminator: max log(D(x)) + log(1 - D(G(Z))
        disc_real = disc(real).reshape(-1) # N x 1 x 1 x 1 = N
        loss_disc_real = loss_func(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = loss_func(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        
        disc.zero_grad() # clear gradient
        loss_disc.backward(retain_graph=True) # backpropagation
        opt_disc.step() # optimize network
        
        # Train Generator: min log(1 - D(G(z)))  <-->  max log(D(G(z)))
        output = disc(fake).reshape(-1) # N x 1 x 1 x 1 = N
        loss_gen = loss_func(output, torch.ones_like(output))
        
        gen.zero_grad() # clear gradient
        loss_gen.backward() # backpropagation
        opt_gen.step() # optimize network
        
        # Print loss & print images to tensorboard
        if i % 10 == 0:
            print("Epoch:",epoch,", Batch:",i,"/",len(dataloader),", Loss of Disc:",loss_disc.item(),", Loss of Gen:",loss_gen.item())
        
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1


# In[6]:


transforms2 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
)
# Produce 50000 fake photos
gen.eval()
with torch.no_grad():
    for i in range(1,50001):
        noise = torch.randn((1,100,1,1)).to(device)
        fake = gen(noise).to("cpu")
        fake = transforms2(fake[0])
        torchvision.utils.save_image(fake, "./fake_imgs/"+str(i)+".png", nrow=1, normalize=True)


# In[ ]:




