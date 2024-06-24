import torch
import numpy as np
import argparse
import sys
import os
import pickle
import time
import utils as uts
from datasets import FashionMNISTDataloaders, getLabelMap, MNISTDataloaders
#from torch.utils.data import Dataset, DataLoader
from network import Network, NetCNN
from loss import Loss
import matplotlib.pyplot as plt
import torch.distributions as tdist
from torch.nn import functional as F
from sklearn.cluster import KMeans
from evaluation import clustering_accuracy
#torch.autograd.set_detect_anomaly(True) # only for debugging
import umap
from plotimg import plot_10figs

parser = argparse.ArgumentParser()
parser.add_argument("dataset",choices=['mnist','fashion'],default='mnist',help="Select datasets")
parser.add_argument("--batch_size",default=256,type=int,help="batch size")
parser.add_argument("--learning_rate",default=3e-4,type=float,help="learning rate configuration")
parser.add_argument("--datapath",default='./data',type=str,help="path/to/dataset")
parser.add_argument("--latent_dim",default=128,type=int,help="dimension of latent features")
parser.add_argument("--epochs",default=10,type=int,help="number of iteration for training")
parser.add_argument("--dca_beta",default=1e-2,type=float,help="trade-off coefficient for privacy funnel")
parser.add_argument("--dca_alpha",default=1e-3,type=float,help="regularization coefficient")
parser.add_argument("--save_dir",default="saved_models",type=str,help="directory to save the model")
parser.add_argument("--cpu",action="store_true",help="force using CPU to run",default=False)
parser.add_argument("--eval_freq",default=5,type=int,help="evaluation frequency, once N epochs")
parser.add_argument("--seed",default=0,type=int,help="random seed number for reproduction")
parser.add_argument("--sampling",action="store_true",default=False,help="random sampling and save the figure")
parser.add_argument("--prior",default="laplace",type=str,help="prior distribution for encoders")
parser.add_argument("--umap_dim",default=2,type=int,help="UMAP dimension for clustering")

args = parser.parse_args()
uts.setup_seed(args.seed)
device = uts.getDevice(args.cpu)

if args.dataset == "fashion":
    train_loader, test_loader = FashionMNISTDataloaders(batch_size=args.batch_size,shuffle=True,device=device,datapath=args.datapath)
elif args.dataset == "mnist":
    train_loader, test_loader = MNISTDataloaders(batch_size=args.batch_size,shuffle=True,device=device,datapath=args.datapath)
else:
    raise NotImplemented("dataset {:} not available".format(args.dataset))

dca_beta = args.dca_beta
dca_alpha = args.dca_alpha

input_shape = (1,28,28) #FIXME: get shape (DATASET)
num_classes = 10 # FIXME: get number of classes (DATASET)
input_size =np.prod(input_shape)
model = Network(input_shape,args.latent_dim,num_classes,device,args.prior).to(device)

uts.print_network(model)
loss_obj = Loss(args.batch_size,num_classes,device).to(device)
optimizer= torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train_alt(epoch):
    tot_loss = 0
    tot_mizy = 0
    tot_entx = 0
    tot_mizx = 0
    tot_eqloss = 0

    bce = torch.nn.BCELoss(reduction="sum")
    mse = torch.nn.MSELoss(reduction="sum")
    ce = torch.nn.CrossEntropyLoss()
    if args.prior == "laplace":
        rz = tdist.Laplace(torch.zeros((1,args.latent_dim),device=device).float(),torch.ones((1,args.latent_dim),device=device).float())
    elif args.prior == "normal":
        rz = tdist.Normal(torch.zeros((1,args.latent_dim),device=device).float(),torch.ones((1,args.latent_dim),device=device).float())
    else:
        raise NotImplementedError("Unsupported prior {:}".format(args.prior))

    for batch, data in enumerate(train_loader):
        x_data, y_label = data
        x_data = x_data.to(device)
        y_label = y_label.to(device)
        batch_size = x_data.size()[0]
        optimizer.zero_grad()
        model.zero_grad()
        # freeze decoder
        model.unfreeze()
        model.freeze_enc()
        xr,z,mu,logvar, qycz = model(x_data)
        # update the encoder...with fitting losses
        loss_list = []
        ent_xcz = bce(xr,x_data)/batch_size
        #ent_xcz = mse(xr,x_data)/batch_size
        loss_list.append(dca_beta * ent_xcz)
        pz = model.pz() # prior learning # Reuse this
        # cross entropy
        y_hard = F.one_hot(y_label,num_classes).float().to(device)
        ce_loss = ce(qycz,y_hard)
        loss_list.append(ce_loss)
        loss = sum(loss_list)
        loss.backward(retain_graph=True)
        # update weights
        optimizer.step()
        tot_loss += loss.item()
        # now update the encoder based on the fitted result
        # second pass
        optimizer.zero_grad()
        model.zero_grad()
        model.unfreeze()
        model.freeze_dec()
        cp_xr,cp_z,cp_mu,cp_logvar,cp_qycz = model(x_data)
        # reconstruct--> estimating H(X)
        cp_rec = bce(cp_xr,x_data)/batch_size # NOTE: BCE
        #cp_rec = mse(cp_xr,x_data)/batch_size # NOTE: MSE
        cp_pz = model.pz()
        
        new_pzcx = model.m_pzcx(*[cp_mu,(0.5*cp_logvar).exp()])
        # leakage --> decoding H(Y|Z)
        new_ce = ce(cp_qycz,y_hard)
        mi_ycz = np.log(num_classes) - new_ce
        
        # privacy funnel update
        diff_pz_kld = loss_obj.kl_divergence(cp_pz,pz,K=1).sum(1).mean()
        # reguarlization
        reg_kld = loss_obj.kl_divergence(new_pzcx,rz,K=1).sum(1).mean()
        #eq_loss = (mi_ycz + diff_pz_kld + dca_beta* cp_bce).abs() + dca_alpha * reg_kld # NOTE: 1-norm version
        eq_loss = 0.5*(mi_ycz + diff_pz_kld + dca_beta* cp_rec).square() + dca_alpha * reg_kld
        eq_loss.backward()
        optimizer.step()
        
        new_mizx = tdist.kl_divergence(new_pzcx,cp_pz).sum(1).mean()
        ent_x = new_mizx + cp_rec
        tot_entx += ent_x.item()
        tot_mizx += new_mizx.item()
        tot_mizy += (np.log(num_classes) + (cp_qycz*cp_qycz.log()).sum(1).mean()).item()
        #
        tot_eqloss += eq_loss.item()
    print("Epoch {:} (train): Loss:{:.6f}, Privacy:{:.6f}, CP_IZY:{:.6f}, CP_IZX:{:.6f}, CP_HX:{:.6f}".format(
        epoch,
        tot_loss/len(train_loader),
        tot_eqloss,
        tot_mizy/len(train_loader),
        tot_mizx/len(train_loader),
        tot_entx/len(train_loader),
        ))


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
    z_gen = []
    
    for batch, data in enumerate(train_loader):
        x_data, y_label = data
        x_data = x_data.to(device)
        with torch.no_grad():
            _,z,_,_,_ = model(x_data)
        z_gen.append(z.detach().cpu())        

    z_gen = torch.cat(z_gen,dim=0).numpy()
    uobj = umap.UMAP(n_components=2)
    z_umap = uobj.fit_transform(z_gen)
    kmeans = KMeans(n_clusters=num_classes,n_init=10)
    kmeans.fit(z_umap)

    z_test = []
    y_true = []
    mse_ac = 0
    num_batches = len(test_loader)
    for batch, data in enumerate(test_loader):
        x_data, y_label = data
        x_data = x_data.to(device)
        with torch.no_grad():
            xr,z,_,_,_ = model(x_data)
        z_test.append(z.detach().cpu())
        y_true.append(y_label)
        mse_x = (xr - x_data).square().flatten(1).mean(1).mean()
        mse_ac += mse_x.item()
    z_test = torch.cat(z_test,dim=0).numpy()
    y_true = torch.cat(y_true,dim=0).numpy()
    z_ut = uobj.transform(z_test)
    y_hat = kmeans.predict(z_ut)
    # label matching
    mse_avg = mse_ac/num_batches
    acc, acc_cnt, tot_cnt = clustering_accuracy(y_true,y_hat)
    print("UMAP EVAL: Accuracy {:.5f}({:}/{:}), MSE:{:.5f}".format(acc,int(acc_cnt),int(tot_cnt),mse_avg))
    return {"acc":acc, "acc_cnt":acc_cnt, "total_cnt":tot_cnt,"mse":mse_avg}

os.makedirs(args.save_dir,exist_ok=True)
rt_dict = {"train_rt":[],"test_rt":[],"ev_metrics":[]} # recorded per epoch
for ep in range(args.epochs):
    tr_t = time.time()
    train_alt(ep)
    tr_dt = time.time() - tr_t
    rt_dict['train_rt'].append(tr_dt)
    if (ep+1)% args.eval_freq == 0:
        ts_t = time.time()
        ev_metrics = test(ep)
        ts_dt = time.time() - ts_t
        rt_dict['test_rt'].append(ts_dt)
        rt_dict['ev_metrics'].append(ev_metrics)
# evaluation phase...
#ev_dict = private_eval()
ev_dict = umap_eval()
# saving the model
fname = "dcaPF_{:}_{:}_ep{:}_bs{:}_lr{:}_ld{:}_beta{:}_alpha{:}_sd{:}".format(
    args.dataset,args.prior,args.epochs,args.batch_size,args.learning_rate,
    args.latent_dim,args.dca_beta,args.dca_alpha,args.seed)

state = model.state_dict()
torch.save(state,os.path.join(args.save_dir,"{:}.pth".format(fname)))
# saving the configuration
result_dict = {"config":vars(args),"runtime":rt_dict,'eval':ev_dict}
with open(os.path.join(args.save_dir,"{:}.pkl".format(fname)),'wb') as fid:
    pickle.dump(result_dict,fid)

# check the reconstruction
labels_map = getLabelMap(args.dataset)
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
        xr_sig = (xr_sig.permute(0,2,3,1)+1)*0.5
        xr_sig = xr_sig.detach().cpu().numpy()
        # for data
        x_sig = (x_sig.permute(0,2,3,1)+1)*0.5
        x_sig = x_sig.detach().cpu().numpy()
        y_sig = y_label.detach().cpu().numpy()
    plot_dict = {"cmap":"gray"} if x_sig.shape[-1] == 1 else {}
    n = int(np.sqrt(nsamp))
    f, ax = plt.subplots(n,n, figsize=(8,8))
    sel_img = np.squeeze(xr_sig[:nsamp])
    sel_img = (sel_img *255).astype("int") # (255-0) byte maps
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np.squeeze(sel_img[i*n+j]),**plot_dict)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title("{:}".format(labels_map[y_sig[i*n+j]]))
    # save, no show
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.savefig("{:}.eps".format(os.path.join(args.save_dir,fname)))

    # plot data
    f, ax = plt.subplots(n,n, figsize=(8,8))
    sel_img = np.squeeze(x_sig[:nsamp])
    sel_img = (sel_img * 255).astype("int") # (255-0) byte maps
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np.squeeze(sel_img[i*n+j]),**plot_dict)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title("{:}".format(labels_map[y_sig[i*n+j]]))
    # save, no show
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.savefig("{:}_data.eps".format(os.path.join(args.save_dir,fname)))
if args.sampling:
    plot_10figs(min(args.batch_size,64))