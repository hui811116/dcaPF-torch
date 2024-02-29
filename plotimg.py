import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_10figs(model,device,dataloader,labels_map,nsamp=64,save_path="./",fname="saved_image"):
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            x_data, y_label = data
            x_data = x_data.to(device)
            xr, _,_,_,_ = model(x_data) 
            xr_sig = xr
            break # for a batch only
        xr_sig = (xr_sig.permute(0,2,3,1) +1 )* 0.5 # (channel last format) rescaling to (0,1)
        xr_sig = xr_sig.detach().cpu().numpy()
        y_sig = y_label.detach().cpu().numpy()
    n = int(np.sqrt(nsamp))
    f, ax = plt.subplots(n,n, figsize=(8,8))
    sel_img = np.squeeze(xr_sig[:nsamp])
    sel_img = (sel_img * 255).astype("int") #(0-255 byte maps)
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np.squeeze(sel_img[i*n+j]))
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title("{:}".format(labels_map[y_sig[i*n+j]]))
    # save, no show
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.savefig(os.path.join(save_path,"{:}.eps".format(fname)))
    f, ax = plt.subplots(n,n, figsize=(8,8))
    sel_img = np.squeeze(xr_sig[:nsamp])
    sel_img = (sel_img * 255).astype("int") #(0-255 byte maps)
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np.squeeze(sel_img[i*n+j]))
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_title("{:}".format(labels_map[y_sig[i*n+j]]))
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.savefig(os.path.join(save_path,"{:}_data.eps".format(fname)))