import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import sys
import numpy as np
import pickle
import pandas as pd
from PIL import Image
#import dlib

def get_img(img_path,transform):
	#img = dlib.load_rgb_image(img_path)
	img = Image.open(img_path).convert("RGB")
	if transform is not None:
		img = transform(img)
	#img = img.view(3,224,224)
	return img

class FairFaceDataset(Dataset):
	def __init__(self,datapath="./data",split='train',transform=None,target_transform=None):
		super().__init__()
		self.transform = transform
		self.target_transform = target_transform
		self.datapath = datapath
		
		if split == "train":
			csv_path = os.path.join(datapath,"fairface_label_train.csv")
			#img_path = os.path.join(datapath,"fairface-img-margin025-trainval/train")
		elif split == "test":
			csv_path = os.path.join(datapath,"fairface_label_val.csv")
			#img_path = os.path.join(datapath,"fairface-img-margin025-trainval/val")
		else:
			raise NotImplementedError("split {:} undefined".format(split))
		self.dataframe = pd.read_csv(csv_path)
		self.img_path = os.path.join(datapath,"fairface-img-margin025-trainval") #NOTE: manual change to margin 1.25 if needed 
		# this is a multi-task dataset
		# file           age  gender        race  service_test
		# age: ['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69','more than 70']
		# gender: ['Male', 'Female']
		# race: ['Black', 'East Asian', 'Indian', 'Latino_Hispanic','Middle Eastern', 'Southeast Asian', 'White']
		self.classes = self.dataframe['race'].to_numpy()
		self.filenames = self.dataframe['file'].to_numpy()
		assert len(self.classes) == len(self.filenames)
		# create label map
		self.label_map = {item:idx for idx, item in enumerate(np.unique(self.classes))}
		self.rev_label_map = {v:k for k,v in self.label_map.items()}
		self.labels = np.array([self.label_map[item] for item in self.classes]).astype("int_")
	def __len__(self):
		return len(self.filenames)
	
	def __getitem__(self,index):
		key = self.filenames[index]
		img_name = os.path.join(self.img_path,key)
		img = get_img(img_name,self.transform)
		label = self.labels[index]
		if self.target_transform is not None:
			label = self.target_transform(label)
		return img, label


def Cifar10Dataset(transform=None,datapath='./data'):
	train_data = datasets.CIFAR10(
		root=datapath,
		train=True,
		download=True,
		transform=transform,
	)
	test_data = datasets.CIFAR10(
		root=datapath,
		train=False,
		download=True,
		transform=transform,
	)
	return train_data, test_data

def FashionMNISTDataloaders(batch_size,shuffle,device,datapath,**kwargs):
	device_dict = {"num_workers":2,**kwargs} if device == "cuda" else {**kwargs}
	test_shuffle= kwargs.get("test_shuffle",False)
	tx = transforms.ToTensor()
	train_set, test_set = FashionMNISTDataset(tx,datapath)
	train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=shuffle,**device_dict)
	test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=test_shuffle,**device_dict)
	return train_loader, test_loader

def FashionMNISTDataset(transform=None,datapath='./data'):
	train_data = datasets.FashionMNIST(
			root=datapath,
			train=True,
			download=True,
			transform=transform,
		)
	test_data = datasets.FashionMNIST(
			root=datapath,
			train=False,
			download=False,
			transform=transform,
		)
	return train_data, test_data

def MNISTDataloaders(batch_size,shuffle,device,datapath,**kwargs):
	device_dict = {"num_workers":2,**kwargs} if device == "cuda" else {**kwargs}
	test_shuffle = kwargs.get('test_shuffle',False)
	tx = transforms.Compose([
			transforms.ToTensor(),
			#transforms.Normalize((0.1307,),(0.3081,))
		])
	train_set ,test_set = MNISTDataset(tx,datapath)
	train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=shuffle,**device_dict)
	test_loader  = DataLoader(test_set,batch_size=batch_size,shuffle=test_shuffle,**device_dict)
	return train_loader, test_loader

def MNISTDataset(transform=None,datapath="./data"):
	train_data = datasets.MNIST(
			root=datapath,
			train=True,
			download=True,
			transform=transform,
		)
	test_data = datasets.MNIST(
			root=datapath,
			train=False,
			download=False,
			transform=transform,
		)
	return train_data, test_data

def synMy():
	gbl_pycx = np.array([[0.90,0.08,0.40],[0.025,0.82,0.05],[0.075,0.10,0.55]])
	gbl_px = np.ones(3,)/3
	gbl_pxy = (gbl_pycx*gbl_px[None,:]).T
	return {'pxy':gbl_pxy,'nx':3, 'ny':3}

def getLabelMap(dtname):
	if dtname =="fashion": # FASHION MNIST Dataset
		label_map = {
				0: 'T-shirt',
				1: 'Trouser',
				2: 'Pullover',
				3: 'Dress',
				4: 'Coat',
				5: 'Sandal',
				6: 'Shirt',
				7: 'Sneaker',
				8: 'Bag',
				9: 'Ankle Boot',
			}
	elif dtname == "mnist": # MNIST DATASET
		label_map = {idx:"{:}".format(idx) for idx in range(10)}
	else:
		raise NotImplemented
	return label_map