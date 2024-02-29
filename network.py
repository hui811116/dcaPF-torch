import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
import numpy as np

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Encoder,self).__init__()
        input_dim = np.prod(input_shape)
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            #nn.Linear(500,500),
            #nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(nn.Linear(500,latent_dim),)
        self.logvar = nn.Sequential(nn.Linear(500,latent_dim),)
    def forward(self, x):
        xx = self.encoder(x)
        mu = self.mu(xx)
        logvar = self.logvar(xx)
        return [mu,logvar]

class Decoder(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(Decoder,self).__init__()
        self.input_shape = input_shape
        input_dim = np.prod(input_shape)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim,500),
            #nn.ReLU(),
            #nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,input_dim),
            nn.Sigmoid(),
        )
    def forward(self,x):
        x = self.decoder(x)
        return x.view(-1,*self.input_shape)

class CNNEncoder(nn.Module):
    def __init__(self,input_shape, latent_dim):
        super(CNNEncoder,self).__init__()
        self.input_shape = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0],16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,4,3,padding=1),
            nn.MaxPool2d(2,2),
            Flatten(),
            #nn.Linear(4*np.prod(input_shape[1:])//16,256),
        )
        flat_size = 4*np.prod(input_shape[1:])//16
        self.mu = nn.Linear(flat_size,latent_dim)
        self.logvar = nn.Linear(flat_size,latent_dim)
    def forward(self,x):
        xx = self.encoder(x)
        mu = self.mu(xx)
        logvar = self.logvar(xx)
        return [mu, logvar]

class CNNDecoder(nn.Module):
    def __init__(self,input_shape, latent_dim):
        super(CNNDecoder,self).__init__()
        self.input_shape = input_shape
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,4*np.prod(input_shape[1:])//16),
            nn.ReLU(),
            Unflatten((4,input_shape[1]//4,input_shape[2]//4)),
            nn.ConvTranspose2d(4,16,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,input_shape[0],2,stride=2),
            nn.Sigmoid(),
        )
    def forward(self,x):
        return self.decoder(x)

class NetCNN(nn.Module):
    def __init__(self,input_shape,feature_dim,num_classes,device,prior):
        super(NetCNN,self).__init__()
        self.enc = CNNEncoder(input_shape,feature_dim).to(device)
        self.dec = CNNDecoder(input_shape,feature_dim).to(device)
        # prior Laplace distribution
        self.pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, feature_dim), requires_grad=False),#mu
            nn.Parameter(torch.zeros(1, feature_dim), requires_grad=True), # logvar,
            #nn.Parameter(torch.zeros(1, feature_dim), requires_grad=False), # logvar,
        ])
        if prior == "laplace":
            self.m_pz = tdist.Laplace # FIXME
            self.m_pzcx = tdist.Laplace # FIXME
        elif prior == "normal":
            self.m_pz = tdist.Normal
            self.m_pzcx = tdist.Normal
        else:
            raise NotImplementedError("unsupported prior {:}".format(prior))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim,num_classes),
            nn.Softmax(dim=1),
        )
    def _pz_params(self):
        # For Laplace, 
        return [self.pz_params[0], self.pz_params[1].exp()]
        #log_processing = self.pz_params[1]
        #return [self.pz_params[0], F.softmax(log_processing-log_processing.amax(keepdim=True),dim=1)+1e-3]
    def pz(self):
        return self.m_pz(*self._pz_params())

    def forward(self,x):
        logits = self.enc(x)
        mu,logvar = logits[0],logits[1]
        #m = tdist.Normal(mu,(0.5*logvar).exp())
        m = self.m_pzcx(*[mu,(0.5*logvar).exp()])
        z = m.rsample()
        xr = self.dec(z)
        qycz = self.classifier(z)
        return xr, z, mu, logvar, qycz
    
    def freeze_enc(self):
        for param in self.enc.parameters():
            param.requires_grad = False
    def freeze_dec(self):
        for param in self.dec.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        # NOTE: making prior always unfreezed
        #for param in self.pz_params.parameters():
        #    param.requires_grad = False
    def unfreeze(self):
        for param in self.enc.parameters():
            param.requires_grad = True
        for param in self.dec.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

###BASIC MLP NETWORK
class Network(nn.Module):
    def __init__(self,input_shape,feature_dim,num_classes,device,prior):
        super(Network,self).__init__()
        self.enc = Encoder(input_shape,feature_dim).to(device)
        self.dec = Decoder(input_shape,feature_dim).to(device)
        # prior Laplace distribution
        self.pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, feature_dim), requires_grad=False),#mu
            nn.Parameter(torch.zeros(1, feature_dim), requires_grad=True), # logvar,
            #nn.Parameter(torch.zeros(1, feature_dim), requires_grad=False), # logvar,
        ])
        if prior == "laplace":
            self.m_pz = tdist.Laplace # FIXME
            self.m_pzcx = tdist.Laplace # FIXME
        elif prior == "normal":
            self.m_pz = tdist.Normal # FIXME
            self.m_pzcx = tdist.Normal # FIXME
        else:
            raise NotImplementedError("undefined prior {:}".format(prior))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim,num_classes),
            nn.Softmax(dim=1),
        )
    def _pz_params(self):
        # For Laplace, 
        return [self.pz_params[0], self.pz_params[1].exp()]
        #log_processing = self.pz_params[1]
        #return [self.pz_params[0], F.softmax(log_processing-log_processing.amax(keepdim=True),dim=1)+1e-3]
    def pz(self):
        return self.m_pz(*self._pz_params())

    def forward(self,x):
        logits = self.enc(x)
        mu,logvar = logits[0],logits[1]
        #m = tdist.Normal(mu,(0.5*logvar).exp())
        m = self.m_pzcx(*[mu,(0.5*logvar).exp()])
        z = m.rsample()
        xr = self.dec(z)
        qycz = self.classifier(z)
        return xr, z, mu, logvar, qycz
    
    def freeze_enc(self):
        for param in self.enc.parameters():
            param.requires_grad = False
    def freeze_dec(self):
        for param in self.dec.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        # NOTE: making prior always unfreezed
        #for param in self.pz_params.parameters():
        #    param.requires_grad = False
    def unfreeze(self):
        for param in self.enc.parameters():
            param.requires_grad = True
        for param in self.dec.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True