"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
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
torch.cuda.device_count()
import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules import Module

from Adam import Adam
from utilities3 import *
import timeit
from timeit import default_timer
torch.backends.cudnn.benchmark = True
# print(torch.cuda.device_count())

torch.manual_seed(0)
np.random.seed(0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description='....')
parser.add_argument('--learning_rate', type=float, default=0.012, help='Learning rate')
parser.add_argument('--num', type=int, default=1, help='num')
parser.add_argument('--len', type=int, default=7, help='len')
parser.add_argument('--modes', type=int, default=12, help='modes')
parser.add_argument('--p', type=int, default=168, help='p')
parser.add_argument('--width', type=int, default=32, help='width')
parser.add_argument('--h', type=int, default=1, help='h')
args = parser.parse_args()
# Gabor_Climate_modeling


global num
global s
global len

num = args.num
s = 28
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


        # 向我们建立的网络module添加新的 parameter
        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("gamma", self.gamma)


    def forward(self):   # input_tensor: 20*32*94*94

        y, x = torch.meshgrid(
            [
                torch.linspace(-0.5, 0.5, s  + len - 1),
                torch.linspace(-0.5, 0.5, s  + len - 1),
            ]
        )

        x = torch.unsqueeze(x,0)
        x = x.repeat(num,1,1)

        y = torch.unsqueeze(y,0)
        y = y.repeat(num,1,1)

        sigma_x = self.sigma 
        sigma_y = sigma_x.float() / (self.gamma+1e-5)

        f = self.freq
        theta = self.theta

        f = f.reshape(num,1,1)
        theta = theta.reshape(num,1,1)
        sigma_x = sigma_x.reshape(num,1,1)
        sigma_y = sigma_y.reshape(num,1,1)


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
        
        return  weight 





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
        self.kernel_num = num

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)) #32,32,12,12
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)) #32,32,12,12

        self.g0 = GaborConv2d(kernel_num = num)
        self.weights3 = nn.Parameter(torch.rand( self.kernel_num, 1)) 

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("kbixy,kioxy->kboxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]   

        gabor =  self.g0()  
        gabor = torch.fft.ifftshift(gabor)

        gabor = torch.unsqueeze(gabor,1)
        gabor = torch.unsqueeze(gabor,1)
        

        gabor1 = gabor[:, :, :, :self.modes1, :self.modes2]
        gabor2 = gabor[:, :, :, -self.modes1:, :self.modes2]
        
        x_ft = torch.fft.rfft2(x)  
                
        x_ft = torch.unsqueeze(x_ft,0)
        x_ft = x_ft.repeat(num,1,1,1,1)        
                
        out_ft = torch.zeros(num, batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)


        weights1 = torch.unsqueeze(self.weights1,0)
        weights1 = weights1.repeat(num,1,1,1,1)
        
        weights2 = torch.unsqueeze(self.weights2,0)
        weights2 = weights2.repeat(num,1,1,1,1)


        out_ft[:,  :, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1 * gabor1)

        out_ft[:, :, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2 * gabor2)


        # test = nn.Softmax(dim=0)(self.weights3)
        weights3 = torch.unsqueeze(nn.Softmax(dim=0)(self.weights3), 2)

        weights3 = torch.unsqueeze(weights3, 3)
        weights3 = torch.unsqueeze(weights3, 4)
        weights3 = weights3.repeat(1, out_ft.shape[1], out_ft.shape[2], out_ft.shape[3], out_ft.shape[4]) 

        x_w = out_ft * weights3  #x_w: 16*20*32*94*94
        out_ft_w = torch.sum(x_w, dim=0)

        #Return to physical space
        x = torch.fft.irfft2(out_ft_w, s=(x.size(-2), x.size(-1)))   # x: 20*32*94*94
                
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
        self.fc0 = nn.Linear(4, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):   # x: 20*85*85*1
        
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)  # x: 20*85*85*3
        
        x = self.fc0(x)  # x: 20*85*85*32
        x = x.permute(0, 3, 1, 2)   # x: 20*32*85*85
        # x = F.pad(x, [0,self.padding, 0,self.padding])   # x: 20*32*94*94

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

        # x = x[..., :-self.padding, :-self.padding]  # x: 20*32*85*85
        x = x.permute(0, 2, 3, 1)  # x: 20*85*85*32
        x = self.fc1(x)  # x: 20*85*85*128
        x = F.gelu(x)
        x = self.fc2(x)  # x: 20*85*85*1

        return x
    


################################################################
# configs
################################################################
ntrain = 60000
ntest = 10000

batch_size = 100
learning_rate = args.learning_rate

epochs = 1000
step_size = 100
gamma = 0.5

modes = 12
width = args.width
h = 28

r = 1
sub = 1
sub1 = 1
s = h
P = args.p
ind = 9
################################################################
# load data and data normalization
################################################################
idxT = [11]
d = np.load("/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/LOCA-main/data/MMNIST_dataset_train.npz")
dispx_allsteps_train = d["dispx_allsteps_train"][:ntrain,idxT,::sub,::sub,None][:,-1,:,:,:]
dispy_allsteps_train = d["dispy_allsteps_train"][:ntrain,idxT,::sub,::sub,None][:,-1,:,:,:]
u_trainx = d["dispx_allsteps_train"][:ntrain,7,::sub,::sub,None]
u_trainy = d["dispy_allsteps_train"][:ntrain,7,::sub,::sub,None]

d = np.load("/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/LOCA-main/data/MMNIST_dataset_test.npz")
dispx_allsteps_test = d["dispx_allsteps_test"][:ntest,idxT,::sub,::sub,None][:,-1,:,:,:]
dispy_allsteps_test = d["dispy_allsteps_test"][:ntest,idxT,::sub,::sub,None][:,-1,:,:,:]
u_testx = d["dispx_allsteps_test"][:ntest,7,::sub,::sub,None]
u_testy = d["dispy_allsteps_test"][:ntest,7,::sub,::sub,None]

S_train = np.concatenate((dispx_allsteps_train,dispy_allsteps_train),axis=-1)
S_test  = np.concatenate((dispx_allsteps_test,dispy_allsteps_test),axis=-1)

U_train = np.concatenate((u_trainx,u_trainy),axis=-1)
U_test  = np.concatenate((u_testx,u_testy),axis=-1)

tag = "CN"
# in_noise_train = 0.15*np.random.normal(loc=0.0, scale=1.0, size=(U_train.shape))
# U_train = U_train + in_noise_train

dtype_double = torch.FloatTensor
cdtype_double = torch.cuda.DoubleTensor
U_train = torch.from_numpy(np.asarray(U_train)).type(dtype_double)
S_train = torch.from_numpy(np.asarray(S_train)).type(dtype_double)

U_test = torch.from_numpy(np.asarray(U_test)).type(dtype_double)
S_test = torch.from_numpy(np.asarray(S_test)).type(dtype_double)

x_train = U_train
y_train = S_train

x_test = U_test
y_test = S_test

###########################################
grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s,s,2), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s,s,2), grid.repeat(ntest,1,1,1)], dim=3)

ind_train = torch.randint(s*s, (ntrain, P))
ind_test = torch.randint(s*s, (ntest, P))
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, ind_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, ind_test), batch_size=batch_size, shuffle=True)

################################################################
# training and evaluation
################################################################

batch_ind = torch.arange(batch_size).reshape(-1, 1).repeat(1, P)

model = FNO2d(modes, modes, width).cuda()
print(count_params(model))



"""
32,28,24,20,16, 12

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



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)


myloss = LpLoss(size_average=False)
start_time = timeit.default_timer()
train_l2_record = []
test_l2_record = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0

    
    for x, y, idx in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s*s,2)
        y = y.reshape(batch_size, s*s,2)
        y = y[batch_ind, idx]
        out = out[batch_ind, idx]

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()


        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y, idx in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s*s,2)
            y = y.reshape(batch_size, s*s,2)
            y = y[batch_ind, idx]
            out = out[batch_ind, idx]

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()

    # print(ep, t2-t1, train_l2, test_l2)#, np.mean(error_total))
    train_l2_record.append(train_l2)
    test_l2_record.append(test_l2)
    
    if ep % 1 == 0 or ep == 999:
        print(ep, '%.2f'% (t2-t1), '%.8f'% train_l2, '\033[1;35m %.8f \033[0m' %test_l2 )

# elapsed = timeit.default_timer() - start_time
# print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)



# ############################################################################################################################


# pred_torch = torch.zeros(S_train.shape)
# baseline_torch = torch.zeros(S_train.shape)
# index = 0
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=1, shuffle=False)
# train_error_u = []
# train_error_u_np = []
# train_error_v_np = []
# with torch.no_grad():
#     for x, y in train_loader:
#         train_l2 = 0
#         x, y= x.cuda(), y.cuda()

#         out = model(x).reshape(1, s, s,2)
#         # out = y_normalizer.decode(out)
#         pred_torch[index,:,:] = out[:,:,:,:]
#         baseline_torch[index,:,:] = y[:,:,:,:]

#         train_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         train_error_u.append(train_l2)
#         train_error_u_np.append(np.linalg.norm(y.cpu().numpy().reshape(S_train.shape[1]*S_train.shape[1],2)[:,0]- out.cpu().numpy().reshape(S_train.shape[1]*S_train.shape[1],2)[:,0],2)/np.linalg.norm(y.cpu().numpy().reshape(S_train.shape[1]*S_train.shape[2],2)[:,0],2))
#         train_error_v_np.append(np.linalg.norm(y.cpu().numpy().reshape(S_train.shape[1]*S_train.shape[1],2)[:,1]- out.cpu().numpy().reshape(S_train.shape[1]*S_train.shape[1],2)[:,1],2)/np.linalg.norm(y.cpu().numpy().reshape(S_train.shape[1]*S_train.shape[2],2)[:,1],2))
#         # print(index, test_l2)
#         index = index + 1

# print("The average train u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_u_np),np.std(train_error_u_np),np.min(train_error_u_np),np.max(train_error_u_np)))
# print("The average train v error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_v_np),np.std(train_error_v_np),np.min(train_error_v_np),np.max(train_error_v_np)))



pred_torch = torch.zeros(S_test.shape)
baseline_torch = torch.zeros(S_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
test_error_u = []
test_error_u_np = []
test_error_v_np = []
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y= x.cuda(), y.cuda()

        out = model(x).reshape(1, s, s,2)
        # out = y_normalizer.decode(out)
        pred_torch[index,:,:] = out[:,:,:,:]
        baseline_torch[index,:,:] = y[:,:,:,:]

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_error_u.append(test_l2)
        test_error_u_np.append(np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,0]- out.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,0],2)/np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[2],2)[:,0],2))
        test_error_v_np.append(np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,1]- out.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,1],2)/np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[2],2)[:,1],2))
        # print(index, test_l2)
        index = index + 1

print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u_np),np.std(test_error_u_np),np.min(test_error_u_np),np.max(test_error_u_np)))
print("The average test v error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_v_np),np.std(test_error_v_np),np.min(test_error_v_np),np.max(test_error_v_np)))





import os
name1 = os.path.basename(__file__).split(".")[0]
name2 = '_l_r_'
name3 = str(args.learning_rate)
torch.save(model, '/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/save_time-upload-to-github/model_save/' + name1 + name2 + name3 +str(args.h))


import scipy.io as io

io.savemat('/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/save_time-upload-to-github/model_save/' + name1 + name2 + name3+str(args.h)+ '.mat', 
           {'train_l2': np.array(train_l2_record), 'test_l2': np.array(test_l2_record)})

print({"finfish"})

io.savemat('/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/save_time-upload-to-github/model_save/' + name1 + '_pred_torch_' +str(args.h) + '.mat', 
           {'pred_torch': np.array(pred_torch)})




# ############################################################################################################################
# #  Add noise
# ##########################################################################################################################

# print("Add noise ")
# sub = 1
# s = 28
# d = np.load("/media/datadisk/Mycode/1.My-DNN-Practice/0.FNO_GaborFilter_Ours/LOCA-main/data/MMNIST_dataset_test.npz")
# dispx_allsteps_test = d["dispx_allsteps_test"][:ntest,idxT,::sub,::sub,None][:,-1,:,:,:]
# dispy_allsteps_test = d["dispy_allsteps_test"][:ntest,idxT,::sub,::sub,None][:,-1,:,:,:]
# u_testx = d["dispx_allsteps_test"][:ntest,7,::sub,::sub,None]
# u_testy = d["dispy_allsteps_test"][:ntest,7,::sub,::sub,None]

# S_test  = np.concatenate((dispx_allsteps_test,dispy_allsteps_test),axis=-1)
# U_test  = np.concatenate((u_testx,u_testy),axis=-1)

# in_noise_test  = 0.15*np.random.normal(loc=0.0, scale=1.0, size=(U_test.shape))
# U_test = U_test + in_noise_test

# dtype_double = torch.FloatTensor
# cdtype_double = torch.cuda.DoubleTensor

# U_test = torch.from_numpy(np.asarray(U_test)).type(dtype_double)
# S_test = torch.from_numpy(np.asarray(S_test)).type(dtype_double)

# x_test = U_test
# y_test = S_test

# grids = []
# grids.append(np.linspace(0, 1, s))
# grids.append(np.linspace(0, 1, s))
# grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
# grid = grid.reshape(1,s,s,2)
# grid = torch.tensor(grid, dtype=torch.float)
# x_test = torch.cat([x_test.reshape(ntest,s,s,2), grid.repeat(ntest,1,1,1)], dim=3)

# pred_torch = torch.zeros(S_test.shape)
# baseline_torch = torch.zeros(S_test.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
# test_error_u = []
# test_error_u_np = []
# test_error_v_np = []
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         x, y= x.cuda(), y.cuda()

#         out = model(x).reshape(1, s, s,2)
#         # out = y_normalizer.decode(out)
#         pred_torch[index,:,:] = out[:,:,:,:]
#         baseline_torch[index,:,:] = y[:,:,:,:]

#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         test_error_u.append(test_l2)
#         test_error_u_np.append(np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,0]- out.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,0],2)/np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[2],2)[:,0],2))
#         test_error_v_np.append(np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,1]- out.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1],2)[:,1],2)/np.linalg.norm(y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[2],2)[:,1],2))
#         # print(index, test_l2)
#         index = index + 1

# print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u_np),np.std(test_error_u_np),np.min(test_error_u_np),np.max(test_error_u_np)))
# print("The average test v error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_v_np),np.std(test_error_v_np),np.min(test_error_v_np),np.max(test_error_v_np)))



