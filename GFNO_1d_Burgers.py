"""
@author: Kai Qi
This file is the Gabor-Filtered Fourier Neural Operator for 1D problem such as the (time-independent) 
Burgers equation discussed in Section 5.3 in the 
[paper](Gabor-Filtered Fourier Neural Operator for Solving Partial Differential Equations).
"""


import argparse
import logging
import math
import operator
from functools import partial, reduce
from re import X
from timeit import default_timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.nn.modules import Conv1d
from Adam import Adam
from utilities3 import *
print(torch.cuda.device_count())
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.benchmark = True

# sub =  1 , s =  8192
# sub =  2 , s =  4096
# sub =  4 , s =  2048
# sub =  8 , s =  1024
# sub = 16 , s =  512
# sub = 32 , s =  256

parser = argparse.ArgumentParser(description='Learning rate')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate')
parser.add_argument('--num', type=int, default =  2,  help='num')
parser.add_argument('--len', type=int, default = 201,  help='len')
parser.add_argument('--ex', type=float, default = 0.998,  help='ex')
parser.add_argument('--epoch', type=int, default = 5000,  help='epoch')
parser.add_argument('--sub', type=int, default = 16,  help='sub')
parser.add_argument('--width', type=int, default=64, help='width')
args = parser.parse_args() 


global num
global len
global ex
global epoch
global size

num = args.num
len = args.len
ex = args.ex
epoch = args.epoch

szie = 2**13 // args.sub



class GaborConv1d(nn.Module):
    def __init__(
        self,
        kernel_num
    ):
        super().__init__()

        self.kernel_num = kernel_num
        
        # self.gamma = nn.Parameter(   torch.tensor([0.08]), requires_grad=True  )
        # self.fwhm = nn.Parameter(    torch.tensor([0.01]), requires_grad=True  )
    
        self.gamma = nn.Parameter(   torch.linspace(0.01, 0.02, self.kernel_num)  , requires_grad=True  )
        self.fwhm = nn.Parameter(   torch.linspace(0.001, 0.002,  self.kernel_num) , requires_grad=True  )
        
        # 向我们建立的网络module添加新的 parameter
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("fwhm", self.fwhm)

    def forward(self):   # input_tensor: 20*32*94*94

        u = torch.linspace(-0.5, 0.5, szie + len - 1).cuda()
        u = torch.unsqueeze(u,0)
        u = u.repeat(self.kernel_num, 1)

        gamma = self.gamma
        fwhm = self.fwhm

        gamma = torch.unsqueeze(gamma,1)
        fwhm = torch.unsqueeze(fwhm,1)
        
        
        test1 = (gamma * torch.pi) / (fwhm +  1e-5)
        test2 =  u - fwhm
        weight = torch.exp( - test1**2 * test2**2) 
        
        
        """
        fig = plt.figure()
        x_plot = torch.linspace(-0.5, 0.5, szie + len - 1)
        plt.subplot(2,1,1)
        plt.plot(x_plot.numpy(), weight[0,:].detach().cpu().numpy())
    
        weight = torch.fft.ifftshift(weight)
        plt.subplot(2,1,2)
        plt.plot(x_plot.numpy(), weight[0,:].detach().cpu().numpy())
        
        plt.show()
        """
        
        return  weight #返回值： 16*20*32*94*94


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels   #64 
        self.out_channels = out_channels   #64
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1， models=16 here.

        self.scale = (1 / (in_channels*out_channels))  #1/(64*64)


        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))  # 64*64*16复数

        self.g0 = GaborConv1d(kernel_num = args.num)
        self.weights2 = nn.Parameter(torch.rand(num, 1)) 
    

    def compl_mul1d(self, input, weights):
        return torch.einsum("kbix,kiox->kbox", input, weights) 


    def forward(self, x):
        # x: 20*64*1024

        batchsize = x.shape[0]
        
        gabor = self.g0()   
        gabor = torch.fft.ifftshift(gabor)


        """
        fig = plt.figure()
        x_plot = torch.linspace(0, szie + len - 1, szie + len - 1)

        plt.plot(x_plot.numpy(), gabor[0,:].detach().cpu().numpy())
        plt.show()
        """
        

        gabor1 = torch.unsqueeze(gabor, 1)
        gabor1 = torch.unsqueeze(gabor1, 1)

        gabor1 = gabor1[:, :, :, :self.modes1]
        
        x_ft = torch.fft.rfft(x)
        x_ft = torch.unsqueeze(x_ft, 0)
        x_ft = x_ft.repeat(num,1,1,1)


        weights1 = torch.unsqueeze(self.weights1,0)
        weights1 = weights1.repeat(num,1,1,1)


        # Multiply relevant Fourier modes
        out_ft = torch.zeros(num, batchsize, self.out_channels, x_ft.shape[3],  device=x.device, dtype=torch.cfloat)  #  40*20*64*513
        
        out_ft[:, :, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :, :self.modes1], weights1 * gabor1)
        
        w = torch.unsqueeze(self.weights2, 2)
        w2 = torch.unsqueeze(w, 3)
        w_ft = w2 * out_ft
        x_ft_final = torch.sum(w_ft, dim=0)

        #Return to physical space
        x = torch.fft.irfft(x_ft_final, n=x.size(-1))  # 20*64*1024


        return x
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes   # 16
        self.width = width    # 64  
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
    
    
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)   
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) #将两个张量（tensor）拼接在一起: (a(x), x)
        x = self.fc0(x)
        # x = self.dropout(x)
        x = x.permute(0, 2, 1)   #20*1024*64
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)  
        x2 = self.w0(x)     
        x = x1 + x2
        x = F.gelu(x)  #20*64*1024

        x1 = self.conv1(x)  
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)  #20*64*1024

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)  #20*64*1024

        x1 = self.conv3(x)

        x2 = self.w3(x)
        x = x1 + x2   #20*64*1024

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)   
        x = self.fc1(x)   #20*1024*128
        # x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc2(x)   #20*1024*1
        # x = self.dropout(x)
        return x    

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################

ntrain = 1000
ntest = 200

# sub = 2**3 #subsampling rate
sub = args.sub #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h   # the resolution

batch_size = 20
learning_rate = args.learning_rate

epochs = epoch
step_size = 100
gamma = 0.5

# modes = 16
modes = 16
width = args.width

################################################################
# read data
################################################################

dataloader = MatReader('/media/datadisk/Mycode/1.My-DNN-Practice/Neural_Operator_Data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]

x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]
# x_test = x_data[1000:1200,:]
# y_test = y_data[1000:1200,:]



x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width).cuda()
print(count_params(model))


"""
[64,52,40,28,16]

model = FNO1d(modes, 64).cuda()
print(count_params(model))
model = FNO1d(modes, 52).cuda()
print(count_params(model))
model = FNO1d(modes, 40).cuda()
print(count_params(model))
model = FNO1d(modes, 28).cuda()
print(count_params(model))
model = FNO1d(modes, 16).cuda()
print(count_params(model))
model = FNO1d(modes, 4).cuda()
print(count_params(model))
"""



################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, ex)


myloss = LpLoss(size_average=False)
train_l2_record = []
test_l2_record = []

from time import *
begin_time = time()


for ep in range(epochs):
    model.train()

    t1 = default_timer()
    train_mse = 0
    train_l2 = 0


    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()  # 对lr进行调整

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= ntrain
    train_l2 /= ntrain     # l2 relative loss
    test_l2 /= ntest       # l2 relative loss

    t2 = default_timer()
    train_l2_record.append(train_l2)
    test_l2_record.append(test_l2)


    if ep % 1 == 0 or ep == 999 or ep == 4999:

        print(ep, '%.2f'% (t2-t1),'%.6f'% train_l2, '\033[1;35m %.8f \033[0m' %test_l2)

    

end_time = time()
run_time = end_time-begin_time



import os
name1 = os.path.basename(__file__).split(".")[0]
name2 = '_sub_'
name3 = str(args.sub)
torch.save(model, 'Gabor/model_save_1/' + name1 + name2 + name3)

import scipy.io as io

io.savemat('Gabor/model_save_1/' + name1 + name2 
           + name3 + '.mat', 
           {'train_l2': np.array(train_l2_record), 'test_l2': np.array(test_l2_record)})

print({"finfish"})