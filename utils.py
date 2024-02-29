import numpy as np
import sys
import os
import torch
from torch import nn
from torch.nn import functional as F
import random
import copy
### torch support
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

def getDevice(force_cpu):
	try:
		if force_cpu:
			device= torch.device("cpu")
			print("force using CPU")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
			print("using Apple MX chipset")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
	except:
		print("MPS is not supported for this version of PyTorch")
		if torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def setup_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True

def getSafeSaveName(savepath,basename,extension=".pkl"):
	repeat_cnt =0
	safename = copy.copy(basename)
	while os.path.isfile(os.path.join(savepath,safename+extension)):
		repeat_cnt += 1
		safename = "{:}_{:}".format(basename,repeat_cnt)
	# return without extension
	return safename

def calcMI(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	return np.sum(pxy*np.log(pxy/px[:,None]/py[None,:]))

def calcEnt(pz):
	return -np.sum(np.log(pz)*pz)

def priorInfo(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T
	return (px,py,pxcy,pycx)

def initPzcx(use_deterministic=0,smooth_val=1e-4,nz=None,nx=None,seed=None):
	rs = np.random.default_rng(seed)
	pzcx = np.zeros((nz,nx))
	if use_deterministic == 1:
		if nz<= nx:
			shuffle_zx = rs.permutation(nz)
			for idx, item in enumerate(shuffle_zx):
				pzcx[item,idx] = 1
			shuffle_rest = rs.integers(nz,size=(nx-nz))
			for nn in range(nx-nz):
				pzcx[shuffle_rest[nn],nz+nn]= 1 
			# smoothing 
			pzcx+= smooth_val
		elif nz-nx==1:
			tmp_pxx = np.eye(nx)
			rng_cols = rs.permutation(nx)
			tmp_pxx = tmp_pxx[:,rng_cols]
			rng_last = rs.permutation(nx)
			last_row = (rng_last==np.arange(nx)).astype("float32")
			pzcx = np.concatenate((tmp_pxx,last_row[None,:]),axis=0)
			pzcx += smooth_val
		else:
			sys.exit("nz is invalid, either >=2 or <= |X|+1")
	else:
		pzcx= rs.random((nz,nx))
	return pzcx / np.sum(pzcx,axis=0,keepdims=True) # normalization