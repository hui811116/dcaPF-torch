import torch
import numpy as np
import argparse
from argparse import Namespace
import sys
import os
import pickle
import time
import utils as uts
from datasets import FashionMNISTDataloaders, getLabelMap, MNISTDataloaders#FairFaceDataset
from network import Network, NetCNN
from loss import Loss
import matplotlib.pyplot as plt
import torch.distributions as tdist
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from evaluation import clustering_accuracy
#torch.autograd.set_detect_anomaly(True) # only for debugging
import umap

parser = argparse.ArgumentParser()
parser.add_argument("load_path",type=str,default="model.pth",help="path/to/pth")
parser.add_argument("--datapath",default='./data',type=str,help="path/to/dataset")
parser.add_argument("--cpu",action="store_true",help="force using CPU to run",default=False)
parser.add_argument("--seed",default=0,type=int,help="random seed number for reproduction")
parser.add_argument("--sampling",action="store_true",default=False,help="random sampling and save the figure")
parser.add_argument("--embedding",action="store_true",default=False,help="generating the projection map")

args = parser.parse_args()

# find the config path
if os.path.isfile(args.load_path):
    cfg_file = ".".join(args.load_path.split(".")[:-1])+".pkl"
    with open(cfg_file,'rb') as fid:
        pkl_dict = pickle.load(fid)
    cfg_dict = pkl_dict['config']
    cfgs = Namespace(**cfg_dict)
else:
    raise RuntimeError("undefined path file {:}".format(args.load_path))


uts.setup_seed(args.seed)
device = uts.getDevice(args.cpu)

if cfgs.dataset == "fashion":
    train_loader, test_loader = FashionMNISTDataloaders(batch_size=cfgs.batch_size,shuffle=True,device=device,datapath=args.datapath)
elif cfgs.dataset == "mnist":
    train_loader, test_loader = MNISTDataloaders(batch_size=cfgs.batch_size,shuffle=True,device=device,datapath=args.datapath)
else:
    raise NotImplemented("dataset {:} not available".format(args.dataset))

rz = tdist.Normal(torch.zeros((1,cfgs.latent_dim),device=device).float(),torch.ones((1,cfgs.latent_dim),device=device).float())
input_shape = (1,28,28) #FIXME: get shape (DATASET)
num_classes = 10 # FIXME: get number of classes (DATASET)
input_size =np.prod(input_shape)
model = Network(input_shape,cfgs.latent_dim,num_classes,device,cfgs.prior).to(device)
model.load_state_dict(torch.load(args.load_path,map_location=device))
model.eval()
uts.print_network(model)

def test(epoch):
    # accuracy of the trained model...
    tot_acc = 0
    tot_cnt = 0
    est_mizx = 0
    est_mizy = 0
    ce = torch.nn.CrossEntropyLoss()
    for batch, data in enumerate(test_loader):
        x_data, y_label = data
        x_data = x_data.to(device)
        with torch.no_grad():
            xr,z,mu,logvar,qycz = model(x_data)
        y_hat = qycz.argmax(dim=1).detach().cpu().numpy()
        tot_acc += np.sum(y_hat == y_label.numpy())
        batch_size = x_data.size()[0]
        tot_cnt += batch_size
        # mutual information estimation
        y_hard = F.one_hot(y_label,num_classes).float().to(device)
        ce_loss = ce(qycz,y_hard)
        est_mizy += (np.log(num_classes) + (qycz*qycz.log()).sum(1).mean()).item()
        pz = model.pz()
        pzcx = model.m_pzcx(*[mu,(0.5*logvar).exp()])
        est_mizx += (tdist.kl_divergence(pzcx,pz).sum(1).mean()).item()
    # calculate metrics
    est_mizx = est_mizx/ len(test_loader)
    est_mizy = est_mizy/ len(test_loader)
    # report
    print("Epoch {:} (TEST), Accuracy:{:.6f}, IZX:{:.6f}, IZY:{:.6f}".format(
        epoch,tot_acc/tot_cnt,est_mizx,est_mizy))
    return {"acc":tot_acc/tot_cnt,"IZX":est_mizx, "IZY":est_mizy}

def umap_eval():
    mse_ac = 0
    num_batches = len(test_loader)
    z_gen = []
    y_true = []
    for batch, data in enumerate(test_loader):
        x_data, y_label = data
        x_data = x_data.to(device)
        with torch.no_grad():
            xr,z,_,_,_ = model(x_data)
        z_gen.append(z.detach().cpu())        
        y_true.append(y_label)
        mse_x = (xr - x_data).square().flatten(1).mean(1).mean()
        mse_ac += mse_x.item()

    z_gen = torch.cat(z_gen,dim=0).numpy()
    uobj = umap.UMAP(n_components=2)
    z_umap = uobj.fit_transform(z_gen)
    kmeans = KMeans(n_clusters=num_classes,n_init=10)
    #y_hat = make_pipeline(StandardScaler(),kmeans).fit_predict(z_umap)
    #kmeans = KMeans(n_clusters=num_classes,n_init=10)
    y_hat = kmeans.fit_predict(z_umap)
    y_true = torch.cat(y_true,dim=0).numpy()
    # label matching
    mse_avg = mse_ac/num_batches
    acc, acc_cnt, tot_cnt = clustering_accuracy(y_true,y_hat)
    print("UMAP EVAL: Accuracy {:.5f}({:}/{:}), MSE:{:.5f}".format(acc,int(acc_cnt),int(tot_cnt),mse_avg))
    return {"acc":acc, "acc_cnt":acc_cnt, "total_cnt":tot_cnt,"mse":mse_avg,"embedding":z_umap,'ytrue':y_true}

ev_metrics = test(0)
# evaluation phase...
ev_dict = umap_eval()
# saving the model
# check the reconstruction
labels_map = getLabelMap(cfgs.dataset)
def plot_10figs(nsamp=64):
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            x_data, y_label = data
            x_data = x_data.to(device)
            xr, _,_,_,_ = model(x_data)
            xr_sig = xr
            x_sig = x_data
            break # for a batch only
        x_sig = x_sig.permute(0,2,3,1).detach().cpu().numpy()
        xr_sig = xr_sig.permute(0,2,3,1).detach().cpu().numpy()
        y_sig = y_label.detach().cpu().numpy()
    plot_dict = {"cmap":"gray"} if x_sig.shape[-1] == 1 else {}
    n = int(np.sqrt(nsamp))
    f, ax = plt.subplots(n,n, figsize=(8,8))
    sel_img = np.squeeze(xr_sig[:nsamp])
    sel_img = ((sel_img+1) * 127.5).astype("int") # (255-0) byte maps
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np.squeeze(sel_img[i*n+j]),**plot_dict)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title("{:}".format(labels_map[y_sig[i*n+j]]))
    # save, no show
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    path_merge = ".".join(args.load_path.split(".")[:-1])
    plt.savefig("{:}.eps".format(path_merge))
    # plot data
    f, ax = plt.subplots(n,n, figsize=(8,8))
    sel_img = np.squeeze(x_sig[:nsamp])
    sel_img = ((sel_img+1) * 127.5).astype("int") # (255-0) byte maps
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np.squeeze(sel_img[i*n+j]),**plot_dict)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title("{:}".format(labels_map[y_sig[i*n+j]]))
    # save, no show
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.savefig("{:}_data.eps".format(path_merge))
if args.sampling:
    plot_10figs(min(cfgs.batch_size,64))


def plot_embeddings(embs,ytrue,label_map):
    plt.figure()
    plt.scatter(embs[:,0],embs[:,1],c=ytrue,s=0.1,cmap="Spectral")
    # add labels
    yset = np.unique(ytrue)
    for item in yset:
        sel_idx = ytrue==item
        m_ex = np.mean(embs[sel_idx,0])
        m_ey = np.mean(embs[sel_idx,1])
        plt.text(m_ex,m_ey,label_map[item],fontsize=16)
    plt.tight_layout()
    path_merge = ".".join(args.load_path.split(".")[:-1])
    plt.savefig("{:}_emb.png".format(path_merge))
    #plt.show()

if args.embedding:
    plot_embeddings(ev_dict['embedding'],ev_dict['ytrue'],labels_map)