import torch
import torch.nn as nn
import numpy as np
from .helper import nxn_cos_sim

def circle_loss(vecs,Y,gamma,m,device):
    
    soft_plus = nn.Softplus()
    Sim = nxn_cos_sim(vecs,vecs)
    
    Sp = np.array([x==Y for x in Y])
    np.fill_diagonal(Sp,False)
    Sn = np.array([x!=Y for x in Y])
    
    Sp = torch.tensor(Sp).to(device)
    Sn = torch.tensor(Sn).to(device)

    Sim_p = torch.masked_select(Sim,Sp).reshape(vecs.shape[0],-1)
    Sim_n = torch.masked_select(Sim,Sn).reshape(vecs.shape[0],-1)
    
    avg_Sp = torch.mean(Sim_p).item()
    avg_Sn = torch.mean(Sim_n).item()

    O_p = 1+m
    O_n = -m
    
    delta_p = 1-m
    delta_n = m
    
    ap = torch.clamp_min(O_p - Sim_p, min=0.)
    an = torch.clamp_min(Sim_n - O_n, min=0.)
    
    logit_p = (-gamma * ap * (Sim_p - delta_p))

    logit_n = (gamma * an * (Sim_n - delta_n))

    loss = soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
    
    return loss,avg_Sp,avg_Sn
    