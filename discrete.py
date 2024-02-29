import numpy as np
import os
import sys
from sklearn.linear_model import Ridge, ElasticNet
from scipy.linalg import block_diag
from scipy.special import softmax,logsumexp
import utils as ut

def ridgePF(pxy,nz,beta,convthres,maxiter,**kwargs):
	#record_flag = kwargs['record']
	record_flag = kwargs.get("record",False)
	alpha_val = kwargs['alpha']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		seed = kwargs.get('seed',None)
		pzcx = ut.initPzcx(0,1e-5,nz,nx,seed) # random init
		#pzcx = ut.initPzcx(1,1e-5,nz,nx,seed) # deterministic init
	itcnt =0
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	conv_flag = False
	# mat kernel
	kernel_mat = pycx.T @ np.linalg.inv(pycx@pycx.T)
	blk_pxcy = block_diag(*([pxcy.T]*nz))
	
	cur_loss = (beta) * ut.calcMI(pzcx@pxy) + ut.calcMI(pzcx*px[None,:])
	while itcnt < maxiter:
		itcnt +=1
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		record_mat[itcnt%record_mat.shape[0]] = cur_loss
		# compute the Czx
		Czx = np.log(est_pz)[:,None] + (1/beta) * np.log(pzcx/est_pz[:,None])
		# stability
		tmp_val = Czx @ kernel_mat 
		raw_pzcy = softmax(tmp_val-np.amax(tmp_val,axis=1,keepdims=True),axis=0)
		raw_pzcy = np.clip(raw_pzcy,a_min=1e-8,a_max=1-1e-8)
		raw_pzcy = raw_pzcy / np.sum(raw_pzcy,axis=0,keepdims=True)
		long_pzcy = raw_pzcy.flatten()
		solver_ridge = Ridge(alpha=alpha_val,positive=True,max_iter=10000,fit_intercept=False)
		solver_ridge.fit(blk_pxcy,long_pzcy)
		
		raw_pzcx_long = solver_ridge.coef_
		raw_pzcx = np.reshape(raw_pzcx_long,(nz,nx))
		# smoothing
		raw_pzcx = np.clip(raw_pzcx,a_min=1e-8,a_max=1-1e-8)
		new_pzcx = raw_pzcx / np.sum(raw_pzcx,axis=0,keepdims=True)
		new_loss = (beta) * ut.calcMI(new_pzcx @ pxy) + ut.calcMI(new_pzcx*px[None,:])
		#print("[LOG] NIT{:} cur_loss:{:.6f}, new_loss{:.6f}".format(itcnt, cur_loss,new_loss))
		if np.fabs(cur_loss - new_loss)< convthres:
			conv_flag = True
			break
		else:
			pzcx = new_pzcx
			cur_loss = new_loss
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx * px[None,:])
	mizy = ut.calcMI(pzcx@pxy)
	output_dict = {"niter":itcnt,"conv":conv_flag,"IZX":mizx,'IZY':mizy,"pzcx":pzcx}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict

def elasticNetPF(pxy,nz,beta,convthres,maxiter,**kwargs):
	record_flag = kwargs['record']
	alpha_val = kwargs['alpha']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(0,1e-5,nz,nx,kwargs['seed']) # random init
	itcnt =0
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	conv_flag = False
	# mat kernel
	kernel_mat = pycx.T @ np.linalg.inv(pycx@pycx.T)
	blk_pxcy = block_diag(*([pxcy.T]*nz))
	
	cur_loss = (beta) * ut.calcMI(pzcx@pxy) + ut.calcMI(pzcx*px[None,:])
	while itcnt < maxiter:
		itcnt +=1
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		record_mat[itcnt%record_mat.shape[0]] = cur_loss
		# compute the Czx
		Czx = np.log(est_pz)[:,None] + (1/beta) * np.log(pzcx/est_pz[:,None])
		# stability
		tmp_val = Czx @ kernel_mat 
		raw_pzcy = softmax(tmp_val-np.amax(tmp_val,axis=1,keepdims=True),axis=0)
		raw_pzcy = np.clip(raw_pzcy,a_min=1e-8,a_max=1-1e-8)
		raw_pzcy = raw_pzcy / np.sum(raw_pzcy,axis=0,keepdims=True)
		long_pzcy = raw_pzcy.flatten()
		solver_ridge = ElasticNet(alpha=alpha_val,positive=True,max_iter=10000,fit_intercept=False)
		solver_ridge.fit(blk_pxcy,long_pzcy)
		
		raw_pzcx_long = solver_ridge.coef_
		raw_pzcx = np.reshape(raw_pzcx_long,(nz,nx))
		# smoothing
		raw_pzcx = np.clip(raw_pzcx,a_min=1e-8,a_max=1-1e-8)
		new_pzcx = raw_pzcx / np.sum(raw_pzcx,axis=0,keepdims=True)
		new_loss = (beta) * ut.calcMI(new_pzcx @ pxy) + ut.calcMI(new_pzcx*px[None,:])
		#print("[LOG] NIT{:} cur_loss:{:.6f}, new_loss{:.6f}".format(itcnt, cur_loss,new_loss))
		if np.fabs(cur_loss - new_loss)< convthres:
			conv_flag = True
			break
		else:
			pzcx = new_pzcx
			cur_loss = new_loss
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx * px[None,:])
	mizy = ut.calcMI(pzcx@pxy)
	output_dict = {"niter":itcnt,"conv":conv_flag,"IZX":mizx,'IZY':mizy,"pzcx":pzcx}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict


def innerSolver(log_pxcy,log_pzcx,c_zcy,**kwargs):
	# compute A_{x|y} p_{z|x} = C_{z|x}
	# subject to: |p_{z|x}|_0 \leq K
	# instead, we compute the problem in log-space
	# var -> logP_{z|x}
	# then \log\sum_{x}\exp{log{P_{x|y}} + log{P_{z|x}}} = B_{y|x}^{\dagger}{\log{p_z^k}+\beta \log{p_{x|z}^k}} (this is C_z|y)
	nz, nx = log_pzcx.shape[0], log_pzcx.shape[1]
	ny = log_pxcy.shape[1]

	expand_log_pxcy = np.repeat(np.expand_dims(log_pxcy,axis=0),nz,axis=0)
	# NOTE: log_pzcx is the step-k solution
	# NOTE: but we need the one that is hidden in logP_{z|y}	
	# transformed problem:
	# minimize |\logsumexp( l_{z|x}+l_{x|y}, axis=1) - C_{z|y}|^2
	# subject to: \max_{z}{\log{p_{z|x}}} > -M (heuristic sparse solution...)
	# FIXME: try \sum{\max{p_{z|x}}} for testing (gradient is equivalent to adding a scaled mask)
	# INNER loop, gradient descent
	# NOTE: this is a convex minimization, should find the optimal with fixed step size
	maxiter = kwargs.get("maxiter",1000)
	stepsize = kwargs.get("ss",1e-3)
	reg_alpha = kwargs.get("alpha",1e-3)
	convthres = kwargs.get("convthres",1e-6)
	reg_mode = kwargs.get("q",2)
	convflag = False
	itcnt =0
	def _compute_loss(l_zcx):
		return 0.5 * np.sum((np.exp(l_zcx)@np.exp(log_pxcy)-softmax(c_zcy,axis=0))**2) -reg_alpha * np.sum(l_zcx)
	def _compute_loss_q1(l_zcx):
		return np.sum(np.fabs((np.exp(l_zcx)@np.exp(log_pxcy)-softmax(c_zcy,axis=0)))) -reg_alpha * np.sum(l_zcx)
	def _grad_q2(expand_log_pzcx):
		est_log_pzcy = logsumexp(expand_log_pxcy+expand_log_pzcx,axis=1,keepdims=True) # (nz,nx,ny)
		softmax_pzcy = softmax(expand_log_pxcy+expand_log_pzcx,axis=1) # (nz,nx,ny)
		return np.sum(est_log_pzcy * softmax_pzcy,axis=2) -reg_alpha # for q=2
	def _grad_q1(expand_log_pzcx):
		est_log_pzcy = logsumexp(expand_log_pxcy+expand_log_pzcx,axis=1,keepdims=True) # (nz,nx,ny)
		sign_grad = np.sign(est_log_pzcy - c_zcy) # (nz,nx,ny)
		softmax_pzcy = softmax(expand_log_pxcy+expand_log_pzcx,axis=1) # (nz,nx,ny)
		return np.sum(sign_grad * softmax_pzcy,axis=1) - reg_alpha
	if reg_mode == 2:
		loss_calc = _compute_loss
		grad_calc = _grad_q2
	elif reg_mode ==1:
		loss_calc = _compute_loss_q1
		grad_calc = _grad_q1
	else:
		raise NotImplementedError("unsupported reg mode {:}".format(reg_mode))
	#start_loss = _compute_loss(log_pzcx)
	start_loss = loss_calc(log_pzcx)
	while itcnt < maxiter:
		itcnt+=1
		# compute required elements
		expand_log_pzcx = np.repeat(np.expand_dims(log_pzcx,axis=2),ny,axis=2) #(nz,nx,ny)	
		#est_log_pzcy = logsumexp(expand_log_pxcy+expand_log_pzcx,axis=1,keepdims=True) # (nz,nx,ny)
		#softmax_pzcy = softmax(expand_log_pxcy+expand_log_pzcx,axis=1) # (nz,nx,ny)
		#max_mask = (log_pzcx == np.amax(log_pzcx,axis=0,keepdims=True)).astype("float32") # (nz,nx)
		# gradient 
		## can be either of the following
		#grad_raw = np.sum(est_log_pzcy * softmax_pzcy,axis=2) -reg_alpha # for q=2
		grad_raw = grad_calc(expand_log_pzcx)

		raw_log_pzcx = log_pzcx - grad_raw * stepsize
		# projection
		raw_log_pzcx -= np.amax(raw_log_pzcx,axis=0,keepdims=True)
		new_log_pzcx = raw_log_pzcx - logsumexp(raw_log_pzcx,axis=0,keepdims=True) # normalized
		# check
		#print(np.sum(np.exp(new_log_pzcx),axis=0))
		#sys.exit()
		# convergence
		new_loss = _compute_loss(new_log_pzcx)
		if np.fabs(new_loss - start_loss) < convthres:
			convflag = True
			break
		else:
			start_loss = new_loss
			log_pzcx = new_log_pzcx
	return {"log_pzcx":log_pzcx,"niter":itcnt,"converge":convflag,"loss":start_loss}

def sparsePF(pxy,nz,beta,convthres,maxiter,**kwargs):
	#record_flag = kwargs['record']
	record_flag = kwargs.get("record",False)
	alpha_val = kwargs['alpha']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		seed = kwargs.get('seed',None)
		pzcx = ut.initPzcx(0,1e-5,nz,nx,seed) # random init
		#pzcx = ut.initPzcx(1,1e-5,nz,nx,seed) # deterministic init
	itcnt =0
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	conv_flag = False
	# mat kernel
	kernel_mat = pycx.T @ np.linalg.inv(pycx@pycx.T)
	#blk_pxcy = block_diag(*([pxcy.T]*nz))
	log_pxcy = np.log(pxcy)
	inner_dict= {"alpha":alpha_val,"ss":5e-3,"maxiter":10000,"convthres":1e-5} # FIXME: connect the arguments
	cur_loss = (beta) * ut.calcMI(pzcx@pxy) + ut.calcMI(pzcx*px[None,:])
	while itcnt < maxiter:
		itcnt +=1
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		record_mat[itcnt%record_mat.shape[0]] = cur_loss
		# compute the Czx
		Czx = np.log(est_pz)[:,None] + (1/beta) * np.log(pzcx/est_pz[:,None])
		# stability
		tmp_val = Czx @ kernel_mat 
		raw_pzcy = softmax(tmp_val-np.amax(tmp_val,axis=1,keepdims=True),axis=0)
		raw_pzcy = np.clip(raw_pzcy,a_min=1e-8,a_max=1-1e-8)
		raw_pzcy = raw_pzcy / np.sum(raw_pzcy,axis=0,keepdims=True)
		#long_pzcy = raw_pzcy.flatten()
		#solver_ridge = Ridge(alpha=alpha_val,positive=True,max_iter=10000,fit_intercept=False)
		#solver_ridge.fit(blk_pxcy,long_pzcy)
		inner_out = innerSolver(log_pxcy,np.log(pzcx),np.log(raw_pzcy),**inner_dict)
		#print("SparsePF debugging: loss:{:.6f}, conv:{:}".format(inner_out['loss'],inner_out['converge']))
		#raw_pzcx_long = solver_ridge.coef_
		#raw_pzcx = np.reshape(raw_pzcx_long,(nz,nx))
		raw_pzcx = np.exp(inner_out['log_pzcx'])
		# smoothing
		raw_pzcx = np.clip(raw_pzcx,a_min=1e-8,a_max=1-1e-8)
		new_pzcx = raw_pzcx / np.sum(raw_pzcx,axis=0,keepdims=True)
		new_loss = (beta) * ut.calcMI(new_pzcx @ pxy) + ut.calcMI(new_pzcx*px[None,:])
		#print("[LOG] NIT{:} cur_loss:{:.6f}, new_loss{:.6f}".format(itcnt, cur_loss,new_loss))
		if np.fabs(cur_loss - new_loss)< convthres:
			conv_flag = True
			break
		else:
			pzcx = new_pzcx
			cur_loss = new_loss
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx * px[None,:])
	mizy = ut.calcMI(pzcx@pxy)
	#print("sparse PF debugging")
	#print(pzcx)
	output_dict = {"niter":itcnt,"conv":conv_flag,"IZX":mizx,'IZY':mizy,"pzcx":pzcx}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict