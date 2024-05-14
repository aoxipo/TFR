import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,in_planes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_planes,planes,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,planes,3,stride,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes,self.expansion*planes,1,bias=False),
            nn.BatchNorm2d(self.expansion*planes),
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self,x):
        identity=x
        out=self.bottleneck(x)
        if self.downsample is not None:
            identity=self.downsample(x)
        out+=identity
        out=self.relu(out)
        return out
 
class FPN(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2]):
        super(FPN,self).__init__()
        self.inplanes=64
        
        self.conv1=nn.Conv2d(in_channel,64,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)
        
        self.layer1=self._make_layer(64,layers[0])
        self.layer2=self._make_layer(128,layers[1],2)
        self.layer3=self._make_layer(256,layers[2],2)
        self.layer4=self._make_layer(512,layers[3],2)
        
        self.toplayer=nn.Conv2d(2048,256,1,1,0)
        
        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)
        
        self.latlayer1=nn.Conv2d(1024,256,1,1,0)
        self.latlayer2=nn.Conv2d(512,256,1,1,0)
        self.latlayer3=nn.Conv2d(256,256,1,1,0)
        
    def _make_layer(self,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes != Bottleneck.expansion * planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,Bottleneck.expansion*planes,1,stride,bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*planes)
            )
        layers=[]
        layers.append(Bottleneck(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*Bottleneck.expansion
        for i in range(1,blocks):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    
    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.upsample(x,size=(H,W),mode='bilinear')+y
    def forward(self,x):
        
        c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)
        
        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))
       
        p4=self.smooth1(p4)
        p3=self.smooth2(p3)
        p2=self.smooth3(p2)
        return p2,p3,p4,p5

class AFPN(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2]):
        super(AFPN,self).__init__()
        self.inplanes=64
        
        self.conv1=nn.Conv2d(in_channel,64,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)
        
        self.layer1=self._make_layer(64,layers[0])
        self.layer2=self._make_layer(128,layers[1],2)
        self.layer3=self._make_layer(256,layers[2],2)
        self.layer4=self._make_layer(512,layers[3],2)
        
        self.toplayer=nn.Conv2d(2048,256,1,1,0)
        
        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)
        
        self.latlayer1=nn.Conv2d(1024,256,1,1,0)
        self.latlayer2=nn.Conv2d(512,256,1,1,0)
        self.latlayer3=nn.Conv2d(256,256,1,1,0)
        
    def _make_layer(self,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes != Bottleneck.expansion * planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,Bottleneck.expansion*planes,1,stride,bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*planes)
            )
        layers=[]
        layers.append(Bottleneck(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*Bottleneck.expansion
        for i in range(1,blocks):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    
    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.upsample(x,size=(H,W),mode='bilinear')+y
        
    def forward(self,x):
        
        c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)
        
        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))

       
        p4=self.smooth1(p4)
        p3=self.smooth2(p3)
        p2=self.smooth3(p2)
        return p2,p3,p4,p5

class CBL(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size = 3, stride = 1, padding = 1):
        super(CBL,self).__init__()
        self.cblblock = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernal_size,stride,padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.cblblock(x)

class MixFpn(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2], num_class = 2, num_require = 1, need_return_dict = True, hidden = 128):
        super(MixFpn,self).__init__()
        self.fpn = FPN(in_channel, layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.hidden = hidden
        self.cbl_same = CBL(self.hidden,self.hidden)
        self.cbl_down1 = CBL(2*self.hidden,self.hidden)
        self.cbl_down = CBL(self.hidden,self.hidden,2,2,0)
        self.conv = nn.Sequential(
            nn.Conv2d(2*self.hidden,64,2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.need_return_dict = need_return_dict
        self.softmax = nn.Linear(4096, 2048)
        self.classifier = nn.Linear(2048, num_class)
    def _upsample(self, x, H, W):
        return F.upsample(x,size=(H,W),mode='bilinear')
    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return self._upsample(x,H,W)+y
    def feature(self, x):
        p2,p3,p4,p5 = self.fpn(x)
        B,C,H,W = p2.shape
        
        for i in range(int(C/self.hidden)-1):
            p2 = self.conv1(p2)
            p3 = self.conv1(p3)
            p4 = self.conv1(p4)
            p5 = self.conv1(p5)
         
        #print(p2.shape, p3.shape,p4.shape,p5.shape,)
        x1 = torch.cat([self.cbl_same(p2), self._upsample(p5,H,W)], dim=1)
        #print(x1.shape)
        feature1 = self.cbl_down1(x1)
        
        x2 = torch.cat([p3, self.cbl_down(feature1)],dim=1)
        #print(x2.shape)
        feature2 = self.cbl_down1(x2)
        
        x3 = torch.cat([p4, self.cbl_down(feature2)],dim=1)
        #print(x3.shape)
        return x1,x2,x3
    def build_results(self,x):
        return {
            "pred_logits":x,
        }
    def forward(self,x):
        x = self.feature(x)
        x = self.conv(x[-1])
        x = x.view(x.shape[0],-1)
        x = self.softmax(x)
        x = self.classifier(x)
        return self.build_results(x) if(self.need_return_dict) else x

class DFPN(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2]):
        super(DFPN,self).__init__()
        self.inplanes=32
        
        self.conv1=nn.Conv2d(in_channel,32,7,2,3,bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)
        
        self.layer1=self._make_layer(32,layers[0])
        self.layer2=self._make_layer(64,layers[1],2)
        self.layer3=self._make_layer(128,layers[2],2)
        self.layer4=self._make_layer(256,layers[3],2)
        
        self.toplayer=nn.Conv2d(1024,256,1,1,0)
        
        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)
        
        self.latlayer1=nn.Conv2d(512,256,1,1,0)
        self.latlayer2=nn.Conv2d(256,256,1,1,0)
        self.latlayer3=nn.Conv2d(128,256,1,1,0)

    def _make_layer(self,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes != Bottleneck.expansion * planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,Bottleneck.expansion*planes,1,stride,bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*planes)
            )
        layers=[]
        layers.append(Bottleneck(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*Bottleneck.expansion
        for i in range(1,blocks):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    
    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.upsample(x,size=(H,W),mode='bilinear')+y
    def forward(self,x):
        
        c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2=self.layer1(c1)
        #print(c2.shape)
        c3=self.layer2(c2)
        #print(c3.shape)
        c4=self.layer3(c3)
        #print(c4.shape)
        c5=self.layer4(c4)
        #print(c5.shape)
        
        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))
       
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2,p3,p4,p5

class EFpn(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2], num_class = 2, num_require = 1, need_return_dict = True, hidden = 128):
        super(EFpn,self).__init__()
        self.fpn = DFPN(in_channel, layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.hidden = hidden
        self.cbl_same = CBL(self.hidden,self.hidden)
        self.cbl_down1 = CBL(2*self.hidden,self.hidden)
        self.cbl_down = CBL(self.hidden,self.hidden,2,2,0)
        self.conv = nn.Sequential(
            nn.Conv2d(2*self.hidden,64,2,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,2,2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.need_return_dict = need_return_dict
        self.softmax = nn.Linear(4096, 2048)
        self.classifier = nn.Linear(2048, num_class)
    def _upsample(self, x, H, W):
        return F.upsample(x,size=(H,W),mode='bilinear')
    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return self._upsample(x,H,W)+y
    def feature(self, x):
        p2,p3,p4,p5 = self.fpn(x)
        return p2
    def build_results(self,x):
        return {
            "pred_logits":x,
        }
    def forward(self,x):
        x = self.feature(x)
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.softmax(x)
        x = self.classifier(x)
        return self.build_results(x) if(self.need_return_dict) else x

# consist
class CDFPN(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2]):
        super(DFPN,self).__init__()
        self.in_channel = in_channel
        self.inplanes=32
        
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(32,layers[0], head = self.multi_head(self.inplanes, 1))    
        self.layer2 = self._make_layer(64,layers[1],2, head = self.multi_head(self.inplanes, 0))  
        self.layer3 = self._make_layer(128,layers[2],2, head = self.multi_head(self.inplanes, 0)) 
        self.layer4 = self._make_layer(256,layers[3],2, head = self.multi_head(self.inplanes, 0)) 
        self.downsample_p2_p3 = nn.Sequential(
            nn.Conv2d(512,64,3,1,1),
            nn.MaxPool2d(2,2),
        )
        self.downsample_p2_p3_p4 = nn.Sequential(
            nn.Conv2d(256+64,64,3,1,1),
            nn.MaxPool2d(2,2),
        ) 
        self.downsample_p2_p3_p4_p5 = nn.Sequential(
            nn.Conv2d(256+64,32,1,1),
        )
        self.p = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8,1024),
            nn.Linear(1024,256),
            nn.Linear(256,2),
        )
        self.p_map = nn.Conv2d(32,1,1,1)
        
        self.toplayer = nn.Conv2d(1024,256,1,1,0)
        
        self.latlayer1 = nn.Conv2d(512,256,1,1,0)
        self.latlayer2 = nn.Conv2d(256,256,1,1,0)
        self.latlayer3 = nn.Conv2d(128,256,1,1,0)

    def multi_head(self, middle_layer = 32,head_deep = 0):
        head_list = [
            nn.Conv2d(self.in_channel, middle_layer, kernel_size=2, stride=2, padding=0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(self.inplanes, eps=1e-5),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ]
        for i in range(head_deep):
            head_list.append(nn.Conv2d(middle_layer, middle_layer, kernel_size=2, stride=2, padding=0, bias=True))
            head_list.append(nn.GELU())
            head_list.append(nn.BatchNorm2d(self.inplanes, eps=1e-5))
            head_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
             
        stem = nn.Sequential(*head_list)
        return stem
        
    def _make_layer(self, planes, blocks, stride=1, head = None):
        downsample=None
        if stride!=1 or self.inplanes != Bottleneck.expansion * planes:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,Bottleneck.expansion*planes,1,stride,bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*planes)
            )
        layers=[]
        if(head is not None):
            layers.append(head)
        layers.append(Bottleneck(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*Bottleneck.expansion
        for i in range(1,blocks):
            layers.append(Bottleneck(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    
    def _upsample_add(self,x,y):
        _,_,H,W = y.shape
        return F.upsample(x,size=(H,W), mode='bilinear') + y
    
    def _resize(self, x):
        return F.upsample(x,size=(H,W), mode='bilinear')
    
    def forward(self, x_512, x_256, x_128, x_64):
       
        c2=self.layer1(x_512)
        #print(c2.shape)
        c3=self.layer2(x_256)
        #print(c3.shape)
        c4=self.layer3(x_128)
        #print(c4.shape)
        c5=self.layer4(x_64)
        #print(c5.shape)
        
        
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5,self.latlayer1(c4))
        p4 = self.relu(p4)
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p3 = self.relu(p3)
        p2 = self._upsample_add(p3,self.latlayer3(c2))
        p2 = self.relu(p2)
       
        p23 = torch.cat([p2,p3], dim = 1)
        p23 = self.downsample_p2_p3(p23)
        
        p234 = torch.cat([p23, p4], dim = 1)
        p234 = self.downsample_p2_p3_p4(p234)

        h_map = torch.cat([p234,p5], dim = 1)
        h_map = self.downsample_p2_p3_p4_p5(h_map)

        return h_map
    
class CEFpn(nn.Module):
    def __init__(self,in_channel = 3, layers = [2,2,2,2], num_class = 2, num_require = 1, need_return_dict = True, hidden = 128):
        super(EFpn,self).__init__()
        self.fpn = DFPN(in_channel, layers)
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8,1024),
            nn.Linear(1024,num_class),
        )
        self.p_map = nn.Conv2d(32,1,1,1)
   
    def build_feature_pyramid(self, x):
        x_1024 = self.downsample(x) # 1024
        x_512 = self.downsample(x_1024)  # 512
        x_256 = self.downsample(x_512)  # 256
        x_128 = self.downsample(x_256)  # 128
        x_64 = self.downsample(x_128)   #  64
        #x_cat = torch.cat([x_512, x_256, x_128, x_64])
        return  x_1024, x_512, x_256, x_128, x_64 # x_cat

    def feature(self, x):
        feature = self.fpn(*x)
        return feature
    def build_results(self,x,y):
        return {
            "pred_logits":x,
            'pred_hot_map':y,
        }
    def forward(self,x):
        x = self.feature(self.build_feature_pyramid(x))
        hot_map = self.p_map(x)
        x = self.classifier(x)
        return self.build_results(x, hot_map) if(self.need_return_dict) else x, hot_map
from dq import ADAE_FPN
class AFPN(nn.Module):
    def __init__(self,
                in_channel = 1, 
                layers = [32, 32, 32, 32], 
                model_dtype = 'big',
                need_return_dict = True
        ):
        super(AFPN,self).__init__()
        self.fpn = ADAE_FPN(
            in_channel = in_channel,
            channel = 64,
            n_res_block = 2,
            n_res_channel = 128,
            n_coder_blocks = 2,
            embed_dim = 64,
            n_codebooks = 2,
            stride = 2,
            decay = 0.99,
            loss_name = "mse",
            vq_type = "dq",
            beta = 0.25,
            n_hier = layers,
            n_logistic_mix = 10,
            model_dtype = model_dtype,
        )

        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(32*8*8,1024),
        #     nn.Linear(1024,num_class),
        # )
        # self.p_map = nn.Conv2d(32,1,1,1)
   
    def build_feature_pyramid(self, x):
        
        x_1024 = self.downsample(x) # 1024
        x_512 = self.downsample(x_1024)  # 512
        x_256 = self.downsample(x_512)  # 256
        x_128 = self.downsample(x_256)  # 128
        x_64 = self.downsample(x_128)   #  64
        # print(x.shape, x_1024.shape, x_512.shape, x_256.shape, x_128.shape, x_64.shape)
        # x_cat = torch.cat([x_512, x_256, x_128, x_64])
        return  x_1024, x_512, x_256, x_128, x_64 # x_cat

    def feature(self, x):
        feature = self.fpn(x)
        return feature
    def build_results(self,x,y):
        return {
            "pred_logits":x,
            'pred_hot_map':y,
        }
    def forward(self, x):
        x, hot_map = self.feature(self.build_feature_pyramid(x))
        # hot_map = self.p_map(x)
        # x = self.classifier(x)
        return self.build_results(x, hot_map) if(self.need_return_dict) else (x, hot_map)


from DesFPN import DesFPN
class DFPN_ (nn.Module):
    def __init__(
                self,
                in_channel = 1,
                block_layers=[3, 6, 12, 8], 
                transition_layer = [128, 288, 560, 568],
                in_channel_layer = [32, 96, 176, 312],
                need_return_dict = True
        ):
        super(DFPN_,self).__init__()
        self.fpn = DesFPN(
            in_channel, 2, growth_rate=32, 
            block_layers=block_layers, 
            transition_layer = transition_layer,
            in_channel_layer = in_channel_layer
        )

        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(32*8*8,1024),
        #     nn.Linear(1024,num_class),
        # )
        # self.p_map = nn.Conv2d(32,1,1,1)
   
    def build_feature_pyramid(self, x):
        x_1024 = self.downsample(x) # 1024
        x_512 = self.downsample(x_1024)  # 512
        x_256 = self.downsample(x_512)  # 256
        x_128 = self.downsample(x_256)  # 128
        # x_64 = self.downsample(x_128)   #  64
        # print(x.shape, x_1024.shape, x_512.shape, x_256.shape, x_128.shape, x_64.shape)
        # x_cat = torch.cat([x_512, x_256, x_128, x_64])
        return  x_1024, x_512, x_256, x_128, # x_64 # x_cat

    def feature(self, x):
        feature = self.fpn(x)
        return feature
    def build_results(self,x,y):
        return {
            "pred_logits":x,
            'pred_hot_map':y,
        }
    def forward(self, x):
        x, hot_map = self.feature(self.build_feature_pyramid(x))
        # hot_map = self.p_map(x)
        # x = self.classifier(x)
        return self.build_results(x, hot_map) if(self.need_return_dict) else (x, hot_map)

def downsample(x):
    x = F.interpolate(x, scale_factor=0.5, mode = 'nearest')
    return x

from DesFPN import DesFPN__
class DFPN__ (nn.Module):
    def __init__(
                self,
                in_channel = 1,
                block_layers=[3, 6, 12, 8], 
                transition_layer = [160, 272, 520, 516],
                in_channel_layer = [16, 80, 136, 260],
                # block_layers=[6, 12, 24, 16], 
                # transition_layer = [256, 512, 1024, 1024],
                # in_channel_layer = [16, 128, 256, 512],
                need_return_dict = True
        ):
        super(DFPN__,self).__init__()
        self.fpn = DesFPN__(
            in_channel, 2, growth_rate=32, 
            block_layers=block_layers, 
            transition_layer = transition_layer,
            in_channel_layer = in_channel_layer
            
        )

        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)
        #self.downsample = downsample
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(32*8*8,1024),
        #     nn.Linear(1024,num_class),
        # )
        # self.p_map = nn.Conv2d(32,1,1,1)
   
    def build_feature_pyramid(self, x):
        x_1024 = self.downsample(x) # 1024
        x_512 = self.downsample(x_1024)  # 512
        x_256 = self.downsample(x_512)  # 256
        x_128 = self.downsample(x_256)  # 128
        # x_64 = self.downsample(x_128)   #  64
        # print(x.shape, x_1024.shape, x_512.shape, x_256.shape, x_128.shape, x_64.shape)
        # x_cat = torch.cat([x_512, x_256, x_128, x_64])
        return  x_1024, x_512, x_256, x_128, # x_64 # x_cat

    def feature(self, x):
        feature = self.fpn(x)
        return feature
    def build_results(self,x,y):
        return {
            "pred_logits":x,
            'pred_hot_map':y,
        }
    def forward(self, x):
        x, hot_map = self.feature(self.build_feature_pyramid(x))
        # hot_map = self.p_map(x)
        # x = self.classifier(x)
        return self.build_results(x, hot_map) if(self.need_return_dict) else (x, hot_map)


from DesFPN import DesFPNC
class DFPNC(DFPN__):
    def __init__(
                self,
                in_channel = 1,
                block_layers=[3, 6, 12, 8], 
                transition_layer = [160, 272, 520, 516],
                in_channel_layer = [16, 80, 136, 260],
                need_return_dict = True
        ):
        super(DFPN__,self).__init__()
        self.fpn = DesFPNC(
            in_channel, 2, growth_rate=32, 
            block_layers=block_layers, 
            transition_layer = transition_layer,
            in_channel_layer = in_channel_layer
            
        )
        self.need_return_dict = need_return_dict
        self.downsample = nn.AvgPool2d(2,2)