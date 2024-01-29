"""
@author: Kai Qi
This file is the Gabor-Filtered Fourier Neural Operator for solving the Darcy Flow equation in Section 5.2.1 in the 
[paper](Gabor-Filtered Fourier Neural Operator for Solving Partial Differential Equations).
"""
import argparse
import math
import operator
import os
from functools import partial, reduce
from timeit import default_timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Adam import Adam
from torch.nn import Parameter
from torch.nn.modules import Module
from utilities3 import *

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='....')
parser.add_argument('--learning_rate', type=float, default=0.036, help='Learning rate')
parser.add_argument('--r', type=int, default=5, help='sample rate for resolution')
parser.add_argument('--modes', type=int, default=12, help='modes')
parser.add_argument('--len', type=int, default=21, help='the length of 2-d Gabor filters in time domain: len * len')
parser.add_argument('--width', type=int, default=32, help='the width in the original FNO')
args = parser.parse_args()

global s
global r
global len
r = args.r
s = int(((421 - 1)/r) + 1)
len  = args.len

class GaborConv2d(Module):
    def __init__(
        self,
        kernel_num
    ):
        super().__init__()

        self.kernel_num = kernel_num
        # small addition to avoid division by zero
        self.delta = 1e-3
        self.freq = nn.Parameter(
            torch.tensor([1.1107]).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = nn.Parameter(
            torch.tensor([0.39]).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = nn.Parameter(torch.tensor([2.82]).type(torch.Tensor), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]).type(torch.Tensor), requires_grad=True)

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("gamma", self.gamma)

    def forward(self):   # The 9 refer to padding size, s+9: data size after padding.
        y, x = torch.meshgrid(
            [
                torch.linspace(-0.5, 0.5, s + 9 + len - 1),
                torch.linspace(-0.5, 0.5, s + 9 + len - 1),
            ]
        )
        sigma_x = self.sigma 
        sigma_y = sigma_x.float() / (self.gamma+1e-5)
        f = self.freq
        theta = self.theta
        u = x.cuda()  * torch.cos(theta) + y.cuda()  * torch.sin(theta)
        v = -x.cuda() * torch.sin(theta) + y.cuda()  * torch.cos(theta)
        test1 = sigma_x**2 *(u- (f/np.pi))**2
        test2 =  sigma_y**2 *  v **2
        weight = torch.exp(-2*np.pi**2 * (test1 + test2))
        return  weight #返回值： 16*20*32*94*94

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)) #32,32,12,12
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)) #32,32,12,12

        self.g0 = GaborConv2d(kernel_num = 1)
        self.weights3 = nn.Parameter(torch.rand(1), requires_grad=True)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]   

        gabor =  self.g0()  
        gabor = torch.fft.ifftshift(gabor)
        gabor = torch.unsqueeze(gabor,0)
        gabor = torch.unsqueeze(gabor,0)
        
        gabor1 = gabor[:, :, :self.modes1, :self.modes2]
        gabor2 = gabor[:, :, -self.modes1:, :self.modes2]
        x_ft = torch.fft.rfft2(x)  

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[ :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1 * gabor1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2 * gabor2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))  
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):   # x: 20*85*85*1
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # x: 20*85*85*3
        x = self.fc0(x)  # x: 20*85*85*32
        x = x.permute(0, 3, 1, 2)   # x: 20*32*85*85
        x = F.pad(x, [0,self.padding, 0,self.padding])   # x: 20*32*94*94

        x1 = self.conv0(x)   
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)   # x: 20*32*94*94

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)  # x: 20*32*94*94

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)  # x: 20*32*94*94

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2   # x: 20*32*94*94

        x = x[..., :-self.padding, :-self.padding]  # x: 20*32*85*85
        x = x.permute(0, 2, 3, 1)  # x: 20*85*85*32
        x = self.fc1(x)  # x: 20*85*85*128
        x = F.gelu(x)
        x = self.fc2(x)  # x: 20*85*85*1

        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
TRAIN_PATH = '/media/datadisk/Mycode/1.My-DNN-Practice/Neural_Operator_Data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = '/media/datadisk/Mycode/1.My-DNN-Practice/Neural_Operator_Data/piececonst_r421_N1024_smooth2.mat'

ntrain = 1000
ntest = 200
batch_size = 20
learning_rate = args.learning_rate
epochs = 1000
step_size = 100
gamma = 0.5
modes = args.modes
width = args.width

r = args.r
h = int(((421 - 1)/r) + 1)
s = h

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH) #coeff: 1024*421*421, sol: 1024*421*421
# test = reader.read_field('coeff')[:ntrain,::r,::r]
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]  #1000*85*85
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]    #1000*85*85

reader.load_file(TEST_PATH)  #coeff: 1024*421*421, sol: 1024*421*421
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]  #200*85*85
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]    #200*85*85

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1) 
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

"""
model = FNO2d(modes, modes, 32).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 28).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 24).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 20).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 16).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 12).cuda()
print(count_params(model))
"""

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)
myloss = LpLoss(size_average=False)
y_normalizer.cuda()
train_l2_record = []
test_l2_record = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    test_l2 /= ntest
    t2 = default_timer()

    if ep%100 == 0 or ep == 999:
        print(ep, '%.2f'% (t2-t1), '%.8f'% train_l2, '\033[1;35m %.8f \033[0m' %test_l2 )
 
    train_l2_record.append(train_l2)
    test_l2_record.append(test_l2)

# import os
# name1 = os.path.basename(__file__).split(".")[0]
# name4 = '_r_'
# name5 = str(args.r)
# torch.save(model, 'Gabor/model_save_1/' + 
#            name1 + name4 + name5)

# import scipy.io as io
# io.savemat('Gabor/model_save_1/' + name1 + name4 + name5 + '.mat', 
#            {'train_l2': np.array(train_l2_record), 'test_l2': np.array(test_l2_record)})
# print({"finfish"})

