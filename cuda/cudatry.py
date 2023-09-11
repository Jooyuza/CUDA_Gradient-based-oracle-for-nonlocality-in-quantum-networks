#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========================== tensor to cuda
# flag = 0
flag = 1
if flag:
    x_cpu = torch.ones((3, 3))
    print("x_cpu:\ndevice: {} is_cuda: {} id: {}".format(x_cpu.device, x_cpu.is_cuda, id(x_cpu)))

    x_gpu = x_cpu.to(device)
    print("x_gpu:\ndevice: {} is_cuda: {} id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))

