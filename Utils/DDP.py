import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .arch import *
from .helper import *
from .loss import *
from .args import *


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()





loss_l=[]
avg_Sp_l=[]
avg_Sn_l=[]
data_dict,length_dict=create_dict(data_folder)

if not os.path.exists(checkpoint_fol):
    os.mkdir(checkpoint_fol)

trf =[
    transforms.ColorJitter(0.85,0.85,0.85,0.5),
    # transforms.RandomAffine(30),
    # transforms.RandomPerspective(),
    transforms.RandomRotation(5),
    transforms.GaussianBlur(7),
    # transforms.RandomErasing(p=1,scale=(0.09,0.18),value='random')
    # transforms.RandomAdjustSharpness(0.5),
    # transforms.RandomAutocontrast()
]
T = transforms.RandomChoice(trf)




def train(rank, world_size):
    
    setup(rank, world_size)
    
    net=Network(rank).to(rank)
    
    net = DDP(net, device_ids=[rank], output_device=rank)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    
    for it in range(iters):
    
        if dist.get_rank() == 0:
            batch,Y = construct_batch(P,K,length_dict)

            batch = [batch[i:i+int(batch_size/world_size)] for i in range(0,batch_size,int(batch_size/world_size))]
            Y = [Y[i:i+int(batch_size/world_size)] for i in range(0,batch_size,int(batch_size/world_size))]

        else:
            batch = [None for _ in range(world_size)]
            Y = [None for _ in range(world_size)]

        inp_batch = [None]
        inp_Y = [None]

        dist.scatter_object_list(inp_batch, batch, src=0)
        dist.scatter_object_list(inp_Y, Y)
        
        dist.barrier()



        lis_np = parse_batch(inp_batch[0],data_dict,img_size)  
        inp = torch.from_numpy(lis_np)
        inp = inp.to(rank)
        inp = inp.permute(0, 3, 1, 2)

        
        inp = torch.stack([T(x) for x in inp])
        
        inp = inp.float()
        
        vecs = net(inp)[0]
        
        loss,avg_Sp,avg_Sn = circle_loss(vecs, inp_Y[0], gamma, margin, rank)
        
        loss = torch.mean(loss)

        if rank ==0:
            print(str(it+1))

        print(rank,'\nloss:',loss,'\nSp:',avg_Sp,'\nSn:',avg_Sn,'\n')
        
        loss_l.append(loss.item())
        avg_Sp_l.append(avg_Sp)
        avg_Sn_l.append(avg_Sn)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if dist.get_rank() == 0:
        
            plt.figure(figsize=(10, 7))
            plt.plot(
                loss_l, color='green', linestyle='-', 
                label='Loss_graph'
            )
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig("loss_plot.png")
            plt.close('all')

            plt.figure(figsize=(10, 7))
            plt.plot(
                avg_Sn_l, color='red', linestyle='-', 
                label='Sn'
            )
            plt.plot(
                avg_Sp_l, color='green', linestyle='-', 
                label='Sp'
            )
            plt.xlabel('Iterations')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig("Sp-Sn_plot.png")
            plt.close('all')
            
            if it%20 ==0 :
                torch.save(net.state_dict(), checkpoint_fol+'/ddp_model_'+str(it)+'.pth')
    
    cleanup()