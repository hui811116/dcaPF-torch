import numpy as np
import os
import sys
import discrete as dc
import utils as ut
import datasets as dt 
import evaluation as ev
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('method',choices=['ridge','sparse'],help="selecting the regularization")
parser.add_argument('--dataset',choices=['syn','heartfail'],default="syn",help="select a discrete dataset")
parser.add_argument('--beta_min',type=float,default=0.01,help="minimum beta for dca")
parser.add_argument('--beta_max',type=float,default=10.0,help="maximum beta for dca")
parser.add_argument('--beta_num',type=int,default=20,help="number of point for dca beta")
parser.add_argument('--alpha_min',type=float,default=0.01,help='minimum alpha for inner dca solver')
parser.add_argument('--alpha_max',type=float,default=10.0,help="maximum alpha for inner dca solver")
parser.add_argument('--alpha_num',type=int,default=16,help="number of point for range of alpha")
parser.add_argument('--nrun',type=int,default=20,help="number of trials per set of hyperparameters")
parser.add_argument('--convthres',type=float,default=1e-6,help="convergence threshold, of the DCA loss")
parser.add_argument('--maxiter',type=int,default=10000,help="the maximum number of iterations")
parser.add_argument("--zmin",type=int,default=2,help="initial search value for the cardinality")
parser.add_argument("--zstep",type=int,default=1,help="search step for the cardinality of the latent variable")
parser.add_argument("--seed",type=int,default=0,help="random seed number for reproduction")
parser.add_argument("--reg_mode",type=int,default=2,help="regularization mode q")
parser.add_argument("--encoder",action="store_true",default=False,help="printing the curve achieving points")

args = parser.parse_args()
# testing the bayes decoder of PF solvers

#beta_range = np.geomspace(0.01,10.0,num=20)
beta_range = np.geomspace(args.beta_min,args.beta_max,num=args.beta_num)
#alpha_range= np.geomspace(0.1,10,num=16)
alpha_range = np.geomspace(args.alpha_min,args.alpha_max,num=args.alpha_num)
#nrun = 20
#conv_thres= 1e-6
#maxiter = 10000

if args.dataset == "syn":
    data = dt.synMy()
elif args.dataset == "heartfail":
    with open('heartfail_pxy.mat','rb') as fid:
        raw_datamat = sio.loadmat(fid)
    gbl_pxy = raw_datamat['heartfail_pxy']
    (nx,ny) = gbl_pxy.shape
    data = {'pxy':gbl_pxy,'nx':nx,'ny':ny}
else:
    raise NotImplementedError("dataset {:} undefined".format(args.dataset))

def calcMI(pxy):
    return np.sum(pxy*np.log(pxy/np.sum(pxy,0,keepdims=True)/np.sum(pxy,1,keepdims=True)))
print("MIXY={:}".format(calcMI(data['pxy'])))
nz_range = np.arange(args.zmin,np.amax(data['pxy'].shape)+2,args.zstep) # include last
#nz_range = [2] # include last
px = np.sum(data['pxy'],axis=1)
#neval = 10000 # FIXME:fixed for fair comparison
res_dicts = []
pf_curve = {}
for nz in nz_range:
    for beta in beta_range:
        for alpha in alpha_range:
            # the best pf loss encoder...
            #best_loss = 0
            #best_dict = {}
            conv_cnt = 0
            update_rate = 0
            for tt in range(args.nrun):
                #alg = dc.ridgePF # FIXME:
                if args.method == "ridge":
                    alg = dc.ridgePF
                elif args.method == "sparse":
                    alg = dc.sparsePF
                else:
                    raise NotImplementedError("undefined method {:}".format(args.method))
                out_dict = alg(data['pxy'],nz,beta,args.convthres,args.maxiter,**{"record":False,"alpha":alpha,"reg_mode":args.reg_mode})
                pf_loss = beta * out_dict['IZY']- out_dict['IZX']
                # computing
                izx_quant = float("{:.2f}".format(out_dict['IZX']))
                est_pz = np.sum(out_dict['pzcx']*px[None,:],axis=1)
                entz = -np.sum(est_pz * np.log(est_pz))
                if not izx_quant in pf_curve.keys():
                    pf_curve[izx_quant] = {'IZY':out_dict['IZY'],'encoder':out_dict['pzcx'],"HZ":entz}
                if out_dict['IZY'] < pf_curve[izx_quant]['IZY']:
                    pf_curve[izx_quant] = {'IZY':out_dict['IZY'],'encoder':out_dict['pzcx'],"HZ":entz}
                    update_rate +=1
                # status
                conv_cnt += int(out_dict['conv'])
                #if tt == 0 or pf_loss < best_loss:
                #    best_loss = pf_loss
                #    best_dict = out_dict
            #ev_xacc, ev_yacc = ev.evalBayes(best_dict['pzcx'],data['pxy'],neval)
            #res_dicts.append({"IZX":best_dict['IZX'],
            #                "IZY":best_dict['IZY'],
            #                "conv":best_dict['conv'],
            #                "niter":best_dict['niter'],
            #                "x_acc":ev_xacc,
            #                "y_acc":ev_yacc,
            #                "beta":beta,
            #                "alpha":alpha})
            # states per avg
            #print("beta,{:.3f},alpha,{:.3f},nz,{:},cv,{:},IZX,{:.3e},IZY,{:.3e},xacc,{:.4f},yacc,{:.4f}".format(
            #    beta,alpha,nz,best_dict['conv'],best_dict['IZX'],best_dict['IZY'],ev_xacc,ev_yacc,
            #))
            print("beta,{:.3f},alpha,{:.3f},nz,{:},cv,{:},npts,{:},updates,{:}".format(
                beta,alpha,nz,conv_cnt/args.nrun,len(pf_curve),update_rate,
            ))

xkey = np.sort(list(pf_curve.keys()))
if args.encoder:
    print("***************")
    print("Achieving Encoders")
    print("***************")
    for item in xkey:
        print("IZX:{:.2f},IZY:{:.4f},HZ:{:.4f}".format(item,pf_curve[item]['IZY'],pf_curve[item]['HZ']))
        print(pf_curve[item]['encoder'])
print("*********")
print("PF CURVE")
print("*********")
for item in xkey:
    print("IZX:{:.2f},IZY:{:.4f}".format(item,pf_curve[item]['IZY']))    
nat2bits = np.log2(np.exp(1))
# xy pairs
plot_xy =[]
#for item in res_dicts:
#    plot_xy.append([item['IZX'],item['IZY'],item['x_acc'],item['y_acc'],item['beta'],item['alpha']])

for item in xkey:
    plot_xy.append([item,pf_curve[item]['IZY']])
plot_xy = np.array(plot_xy)
plt.scatter(plot_xy[:,0],plot_xy[:,1])
plt.show()

