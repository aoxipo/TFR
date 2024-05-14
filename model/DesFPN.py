
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.LeakyReLU(in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.LeakyReLU(in_channel),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer
    
class DesFPN(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, 
                block_layers=[3, 6, 12, 8], 
                transition_layer = [128, 288, 560, 568],
                in_channel_layer = [32, 96, 176, 312]
         
        ):
        super(DesFPN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 4, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 4, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 4, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 4, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.DB1 = self._make_dense_block(in_channel_layer[0], growth_rate, num=block_layers[0])
        self.TL1 = self._make_transition_layer(transition_layer[0])
        self.DB2 = self._make_dense_block(in_channel_layer[1], growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(transition_layer[1])
        self.DB3 = self._make_dense_block(in_channel_layer[2], growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(transition_layer[2])
        self.DB4 = self._make_dense_block(in_channel_layer[3], growth_rate, num=block_layers[3])
        self.TL4 = self._make_transition_layer(transition_layer[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(transition_layer[-1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(transition_layer[-1], num_classes)
        self.hot_map = nn.Sequential(
            nn.Conv2d(transition_layer[-1], 64,2,2),
            nn.Conv2d(64,1,1,1),
        )
        #self.ea = nn.Linear(1024,2)
 
    def forward(self, x):
        # input 1024,1024  512,512  256,256 128,128  64,64 
        x_1024, x_512, x_256, x_128 = x[0], x[1], x[2], x[3]
        
        x_1024 = self.block1(x_1024)
        x_512 = self.block2(x_512)
        x_256 = self.block3(x_256)
        x_128 = self.block4(x_128)
        #x_64 = self.block1(x_64)
        #print(x_1024.shape, x_512.shape, x_256.shape, x_128.shape, x_64.shape)
        
        #         print("orgin:",x.size())
        #         x = self.block1(x)
        #         print("block1:",x.size())
        x = self.DB1(x_1024)
        #print("DB1:",x.size())
        x = self.TL1(x)
        #print("TL1:",x.size())
        
        x = torch.cat([x, x_512], dim = 1)
        #print('512 cat: ', x.shape)
        x = self.DB2(x)
        #print("DB2:",x.size())
        x = self.TL2(x)
        #print("TL2:",x.size())
        
        x = torch.cat([x, x_256], dim = 1)
        #print('256 cat: ', x.shape)
        x = self.DB3(x)
        #print("DB3:",x.size())
        x = self.TL3(x)
        #print("TL3:",x.size())
        
        x = torch.cat([x, x_128], dim = 1)
        #print('128 cat: ', x.shape)
        x = self.DB4(x)
        #print("DB4:",x.size())
        data_map = self.hot_map(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        #print(x.size())
        #a = self.ea(x)
        #print(a.size())
        x = self.classifier(x)
        #print(x.size())
        return x, data_map

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)
    
class ConvOr(nn.Module):
    def __init__(self,in_channel, need = True):
        super().__init__()
        self.need = need
        n = []
        if need > 0:
            for i in range(abs(need)):
                n.append(nn.Conv2d(in_channel, in_channel, 2, 2))
            self.conv = nn.Sequential(*n)
        elif need < 0:
            for i in range(abs(need)):
                n.append(nn.ConvTranspose2d(in_channel, in_channel, 2, 2))
            self.conv = nn.Sequential(*n)
    def forward(self, x):
        if self.need != 0:
            return self.conv(x)
        return x
    
class merge_block(nn.Module):
    def __init__(self, in_channel, base = [0,0,0,0]):
        super().__init__()
        n = []
        for i in range(4):
            n.append(ConvOr(in_channel, base[i]))
        self.net = nn.Sequential(*n)
    def forward(self, x_128, x_64, x_32, x_16):
        x = []
        x.append(self.net[0](x_128))
        x.append(self.net[1](x_64))
        x.append(self.net[2](x_32))
        x.append(self.net[3](x_16))
        x = torch.cat(x, 1)
        return x
    
class DesFPN_(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16], transition_layer = [256,512,1024], in_channel_layer = [64,128,256,512],need_return_dict = True):
        super().__init__()
        self.need_return_dict = need_return_dict
        
        block = [
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 2, 3, bias = False),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channel_layer[0], in_channel_layer[0], 7, 2, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        ]
        
        self.merge1 = merge_block(in_channel_layer[0], [0, -1, -2, -3])
        # self.merge2 = merge_block(in_channel_layer[0], [1, 0, -1, -2])
        # self.merge3 = merge_block(in_channel_layer[0], [2, 1, 0, -1])
        # self.merge4 = merge_block(in_channel_layer[0], [3, 2, 1, 0])
        
        self.block1 = nn.Sequential(*block)
        self.block2 = nn.Sequential(*block)
        self.block3 = nn.Sequential(*block)
        self.block4 = nn.Sequential(*block)
        
        self.DB1 = self._make_dense_block(4 * in_channel_layer[0], growth_rate, num=block_layers[0])
        self.TL1 = self._make_transition_layer(transition_layer[0])
        self.DB2 = self._make_dense_block(in_channel_layer[1], growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(transition_layer[1])
        self.DB3 = self._make_dense_block(in_channel_layer[2], growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(transition_layer[2])
        self.DB4 = self._make_dense_block(in_channel_layer[3], growth_rate, num=block_layers[3])
        self.TL4 = self._make_transition_layer(transition_layer[3])
        self.global_average = nn.Sequential(
            #nn.BatchNorm2d(1),
            nn.Conv2d(transition_layer[-1], 1024, 2, 2),
            nn.Conv2d(1024, 1024, 2, 2),
            nn.Conv2d(1024, 1024, 2, 2),
            nn.Flatten(),
            nn.Linear(4096,2),
        )
        #self.classifier = nn.Linear(2, num_classes)
        self.hot_map = nn.Sequential(
            nn.Conv2d(transition_layer[-1], 256,2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,1,1,1),
        )
        #self.ea = nn.Linear(1024,2)
    def build_results(self,x):
        return {
            "pred_logits":x,
        }
    def forward(self, x_1024, x_512, x_256, x_128):

        x_1024 = self.block1(x_1024)
        x_512 = self.block2(x_512)
        x_256 = self.block3(x_256)
        x_128 = self.block4(x_128)
      
        #print(x_1024.shape, x_512.shape, x_256.shape, x_128.shape)
        x = torch.cat(x_1024, x_512, x_256, x_128, 1)
        #print(x.shape)
        x = self.DB1(x)
        #print("DB1:",x.size())
        x = self.TL1(x)
        #print("TL1:",x.size())
        
     
        
        #print('512 cat: ', x.shape)
        x = self.DB2(x)
        #print("DB2:",x.size())
        x = self.TL2(x)
        #print("TL2:",x.size())
        
        #print('256 cat: ', x.shape)
        x = self.DB3(x)
        #print("DB3:",x.size())
        x = self.TL3(x)
        #print("TL3:",x.size())
        
        #print('128 cat: ', x.shape)
        x = self.DB4(x)
        #print("DB4:",x.size())
        hotmap= self.hot_map(x)
        #print(hotmap.shape)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        #print(x.size())
      
        return x, hotmap

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)
    

class DesFPN__(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, 
                block_layers=[6, 12, 24, 16], 
                transition_layer = [256, 512, 1024, 1024],
                in_channel_layer = [16, 128, 256, 512],
                
                need_return_dict = True):
        super(DesFPN__, self).__init__()
        self.need_return_dict = need_return_dict
        
        block = [
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 2, 3, bias = False),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channel_layer[0], in_channel_layer[0], 7, 2, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        ]
        
        self.block1 = nn.Sequential(*block)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 7, 2, 3),
            nn.BatchNorm2d(in_channel_layer[0]),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel_layer[0], 1, 1),
        )
        
        self.DB1 = self._make_dense_block(4 * in_channel_layer[0], growth_rate, num=block_layers[0])
        self.TL1 = self._make_transition_layer(transition_layer[0])
        self.DB2 = self._make_dense_block(in_channel_layer[1], growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(transition_layer[1])
        self.DB3 = self._make_dense_block(in_channel_layer[2], growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(transition_layer[2])
        self.DB4 = self._make_dense_block(in_channel_layer[3], growth_rate, num=block_layers[3])
        self.TL4 = self._make_transition_layer(transition_layer[3])
        self.global_average = nn.Sequential(
       
            nn.Conv2d(transition_layer[-1], 1024, 2, 2),
            nn.Conv2d(1024, 1024, 2, 2),
            nn.Conv2d(1024, 1024, 2, 2),
            nn.Flatten(),
            nn.Linear(4096, num_classes),
        )
        self.hot_map = nn.Sequential(
            nn.Conv2d(transition_layer[-1], 256,2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,1,1,1),
        )
        #self.ea = nn.Linear(1024,2)
    def forward(self, x):
        x_1024, x_512, x_256, x_128 = x[0], x[1], x[2], x[3]
        x_1024 = self.block1(x_1024)
        x_512 = self.block2(x_512)
        x_256 = self.block3(x_256)
        x_128 = self.block4(x_128)
      
        #print(x_1024.shape, x_512.shape, x_256.shape, x_128.shape)
        x = torch.cat([x_1024, x_512, x_256, x_128], 1)
        #print(x.shape)
        x = self.DB1(x)
        #print("DB1:",x.size())
        x = self.TL1(x)
        #print("TL1:",x.size())
        
     
        
        #print('512 cat: ', x.shape)
        x = self.DB2(x)
        #print("DB2:",x.size())
        x = self.TL2(x)
        #print("TL2:",x.size())
        
        # print('256 cat: ', x.shape)
        x = self.DB3(x)
        # print("DB3:",x.size())
        x = self.TL3(x)
        # print("TL3:",x.size())
        
        # print('128 cat: ', x.shape)
        x = self.DB4(x)
        # print("DB4:",x.size())
        hotmap= self.hot_map(x)
        # print(hotmap.shape)
        x = self.global_average(x)
        # print(x.size())
      
        return x, hotmap

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)
    
from utils import crop_tensor
class DesFPNC(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, 
                 block_layers=[6, 12, 24, 16], 
                 transition_layer = [256,512,1024], 
                 in_channel_layer = [64,128,256,512],
        need_return_dict = True):
        super(DesFPNC, self).__init__()
        self.need_return_dict = need_return_dict
        # 
        # block = [
        #     nn.Conv2d(in_channel, in_channel_layer[0], 7, 2, 3, bias = False),
        #     nn.BatchNorm2d(in_channel_layer[0]),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channel_layer[0], in_channel_layer[0], 7, 2, 3),
        #     nn.BatchNorm2d(in_channel_layer[0]),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(3, 2, padding=1)
        # ]
        
        # #         self.merge1 = merge_block(in_channel_layer[0], [0, -1, -2, -3])
        # #         self.merge2 = merge_block(in_channel_layer[0], [1, 0, -1, -2])
        # #         self.merge3 = merge_block(in_channel_layer[0], [2, 1, 0, -1])
        # #         self.merge4 = merge_block(in_channel_layer[0], [3, 2, 1, 0])
        # self.block1 = nn.Sequential(*block)
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel_layer[0], 7, 2, 3),
        #     nn.BatchNorm2d(in_channel_layer[0]),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(3, 2, padding=1)
        # )
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel_layer[0], 2, 2),
        # )
        # # self.preconv = nn.Conv2d(3 * in_channel_layer[0], 4 * in_channel_layer[0], 1, 1)
        # self.block4 = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel_layer[0], 1, 1),
        # )
        
        self.DB1 = self._make_dense_block(in_channel_layer[0], growth_rate, num=block_layers[0])
        self.TL1 = self._make_transition_layer(transition_layer[0])
        self.DB2 = self._make_dense_block(in_channel_layer[1], growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(transition_layer[1])
        self.DB3 = self._make_dense_block(in_channel_layer[2], growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(transition_layer[2])
        self.DB4 = self._make_dense_block(in_channel_layer[3], growth_rate, num=block_layers[3])
        self.TL4 = self._make_transition_layer(transition_layer[3])
        self.global_average = nn.Sequential(
            #nn.BatchNorm2d(1),
            nn.Conv2d(transition_layer[-1], 1024, 2, 2),
            nn.Conv2d(1024, 1024, 2, 2),
            nn.Conv2d(1024, 1024, 2, 2),
            nn.Flatten(),
            nn.Linear(4096,2),
        )
        #self.classifier = nn.Linear(2, num_classes)
        self.hot_map = nn.Sequential(
            nn.Conv2d(transition_layer[-1], 256,2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,1,1,1),
        )
    def get_mask(self, mask):
        b = nn.functional.interpolate(mask, (1024, 1024), mode='bicubic')
        thera = torch.max(b)/4
        # thera = 1
        # print(thera)
        b[b<thera] = 0
        b[b>thera] = 1
        return b
    def forward(self, x_1024, x_512, x_256, x_128):
        # total = x_1024 * x_512 * x_256 * x_128
        # get_mask()
        #x_1024_small = self.block1(x_1024)
        x_1024 = crop_tensor(x_1024,8)
        
        x_512 = crop_tensor(x_512,4)
        x_256 = crop_tensor(x_256,2)
        x_128 = x_128.unsqueeze(1)
        #
        #x_512 = self.block2(x_512)
        #x_256 = self.block3(x_256)
        #x_128 = self.block4(x_128)
      
        # print(x_1024.shape, x_512.shape, x_256.shape, x_128.shape)
        x = torch.cat([x_1024, x_512, x_256, x_128], 1)
        x = x.squeeze(2)
        # x = self.preconv(x)
        # print(x.shape)
        x = self.DB1(x)
        # print("DB1:",x.size())
        x = self.TL1(x)
        # print("TL1:",x.size())
        
     
        
        # print('512 cat: ', x.shape)
        x = self.DB2(x)
        # print("DB2:",x.size())
        x = self.TL2(x)
        # print("TL2:",x.size())
        
        # print('256 cat: ', x.shape)
        x = self.DB3(x)
        # print("DB3:",x.size())
        x = self.TL3(x)
        # print("TL3:",x.size())
        
        # print('128 cat: ', x.shape)
        x = self.DB4(x)
        # print("DB4:",x.size())
        hotmap = self.hot_map(x)
        # print(hotmap.shape)
        x = self.global_average(x)
        #x = x.view(x.shape[0], -1)
        # print(x.size())
      
        return x, hotmap

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)



