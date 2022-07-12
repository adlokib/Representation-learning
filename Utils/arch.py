from .helper import pos_embed_2d
import torch.nn as nn
import torchvision

class Network(nn.Module):
    
    def __init__(self,device):
        
        super().__init__()

        self.dims = 128
        
        self.backbone = nn.Sequential(*(list(torchvision.models.resnet18(pretrained=True).children())[:-2]))

        ## If backboke has to be partially frozen
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # for child in list(self.backbone.children())[-4:]:
        #     for param in child.parameters():
        #         param.requires_grad = True
        
        self.conv = nn.Conv2d(in_channels=512, out_channels=self.dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.dims)

        self.pos_mat = pos_embed_2d((8,16),self.dims).to(device)

        self.encoder_layer1 = nn.TransformerEncoderLayer(self.dims,4,self.dims*4,batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(self.dims,4,self.dims*4,batch_first=True)
        self.encoder_layer3 = nn.TransformerEncoderLayer(self.dims,4,self.dims*4,batch_first=True)
        
        self.fc = nn.Linear(in_features=self.dims * 8 * 16, out_features=self.dims)

    
    def branch(self,x):
        
        t = self.backbone(x)

        t = self.conv(t)
        t = self.bn(t)
       
        t = t.permute(0,2,3,1)

        inp = self.pos_mat + t
        inp = inp.reshape(-1,8*16,self.dims) 
        
        t = self.encoder_layer1(inp)
        t = self.encoder_layer2(t)
        t = self.encoder_layer3(t)
        
        t = t.reshape(-1, self.dims * 8 * 16)
        
        t = self.fc(t)
        
        return t
    
    
    def forward(self, *kwargs):
        
        vecs =[]
        for inp in kwargs:
            vecs.append(self.branch(inp))
        
        return vecs
