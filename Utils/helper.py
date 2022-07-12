import torch
import glob,random
import numpy as np
import cv2


def pos_embed_2d(shape_2d,dims):
    x = shape_2d[1]
    y = shape_2d[0]
    quart = int(dims/4)
    
    pos_mat = torch.zeros(y,x,dims)
    
    for i in range(y):
        for j in range(x):
            for k in range(quart):
                pos_mat[i,j,2*k] = np.sin(j/(10000**((4*k)/dims)))
                pos_mat[i,j,2*k+1] = np.cos(j/(10000**((4*k)/dims)))
                pos_mat[i,j,2*k+2*quart] = np.sin(i/(10000**((4*k)/dims)))
                pos_mat[i,j,(2*k+1)+(2*quart)] = np.cos(i/(10000**((4*k)/dims)))
    
    return pos_mat



def construct_batch(P,K,length_dict):
    classes = [(x,i) for i,x in enumerate(list(length_dict.keys()))]
    
    random.shuffle(classes)
    
    clss = classes[:P]
    
    batch=[]
    Y=[]
    for cls in clss:
        
        all_data_points = [x for x in range(length_dict[cls[0]])]
        random.shuffle(all_data_points)
        selected_data_points = all_data_points[:K]
                           
        
        batch.extend([(cls[0],x) for x in selected_data_points])
        Y.extend([cls[1] for _ in range(K)])
        
    return batch,np.array(Y)



def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)



def create_dict(folders):
    classes=[]
    for fol in folders:
        classes.extend(glob.glob(fol+'/*'))
    
    data_dict={}
    length={}
    
    for cls in classes:
        data_dict[cls]=glob.glob(cls+'/*')
        length[cls]=len(data_dict[cls])
        
    return data_dict,length



def parse_batch(batch,data_dict,img_size):

    array_l= [] 
    
    for entry in batch:
        img_path=data_dict[entry[0]][entry[1]]
        img=cv2.imread(img_path)
        if img.shape[:2]!=img_size:
            img=cv2.resize(img,(img_size[1],img_size[0]))
        array_l.append(img)

    array_l=np.array(array_l)

    return array_l
            





