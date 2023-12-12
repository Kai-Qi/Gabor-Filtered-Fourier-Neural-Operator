"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

import argparse
import math
import operator
from functools import partial, reduce
from timeit import default_timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

from Adam import Adam
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)
from timeit import default_timer

from Adam import Adam
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='....')
parser.add_argument('--learning_rate', type=float, default=0.021, help='Learning rate')
parser.add_argument('--len', type=int, default=11, help='len')
parser.add_argument('--size_frequency', type=int, default=74, help='size_frequency')
parser.add_argument('--width', type=int, default=20, help='width')

args = parser.parse_args()



global size_frequency
size_frequency = args.size_frequency
s = 65




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


        # 向我们建立的网络module添加新的 parameter
        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("gamma", self.gamma)


    def forward(self):   # input_tensor: 20*32*94*94

        y, x = torch.meshgrid(
            [
                torch.linspace(-0.5, 0.5, size_frequency),
                torch.linspace(-0.5, 0.5, size_frequency),
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
        
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1, projection='3d')

        ax1.plot_surface(x.detach().cpu().numpy(),
                         y.detach().cpu().numpy(),
                         weight.detach().cpu().numpy(),
                         cmap="rainbow")
        plt.show()
        """
        
        return  weight #返回值： 16*20*32*94*94



################################################################
# fourier layer
################################################################
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

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
        
        # index_1 = int( (s + 9 + len - 1  - 24)/2 )
        # index_2 = index_1 + 12
        # index_3 = index_2 + 12

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
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm2d(self.width)
        # self.bn1 = torch.nn.BatchNorm2d(self.width)
        # self.bn2 = torch.nn.BatchNorm2d(self.width)
        # self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
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

 

# TRAIN_PATH = '/media/datadisk/Mycode/1.My-DNN-Practice/Neural_Operator_Data/ns_V1e-3_N5000_T50.mat'


ntrain = 1000
ntest = 200

modes = 12
width = args.width

batch_size = 20
batch_size2 = batch_size

epochs = 1000
learning_rate = args.learning_rate
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

# path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# path_model = 'model/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path

# sub = 1
# S = 64
# T_in = 10
# T = 40
# step = 1

################################################################
# load data
################################################################

TRAIN_PATH = '/media/datadisk/Mycode/1.My-DNN-Practice/Neural_Operator_Data/NavierStokes_V1e-5_N1200_T20.mat'
sub = 1
S = 64
T_in = 10
T = 10
step = 1

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)


################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')
print(count_params(model))



"""
[20,18,16,14,12,10]

model = FNO2d(modes, modes, 20).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 18).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 16).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 14).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 12).cuda()
print(count_params(model))
model = FNO2d(modes, modes, 10).cuda()
print(count_params(model))

"""


optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)


train_l2_record = []
test_l2_record = []
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0


    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()


        optimizer.step()
    scheduler.step()
    test_l2_step = 0
    test_l2_full = 0
    if ep%1 ==0 or ep==999:
        with torch.no_grad():
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        t2 = default_timer()

        # print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
        #       test_l2_full / ntest)

        print(ep, '%.2f'% (t2-t1), '%.6f'% (train_l2_step / ntrain / (T / step)), '%.6f'% (train_l2_full / ntrain), '%.6f'% (test_l2_step / ntest / (T / step)), \
            '\033[1;35m %.6f \033[0m' % (test_l2_full / ntest))

    train_l2_record.append(train_l2_full / ntrain)
    test_l2_record.append(test_l2_full / ntest)




# import os
# name1 = os.path.basename(__file__).split(".")[0]

# name2 = "_l_"
# name3 = str(args.l)

# torch.save(model, '/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/JCP_revise/'+ name1+ name2+ name3  )


# import scipy.io as io

# io.savemat('/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/JCP_revise/' + name1+ name2+ name3 + '.mat', 
#            {'train_l2': np.array(train_l2_record), 'test_l2': np.array(test_l2_record)})


# print('finish!')