import torch
import torch.nn as nn
import torch.nn.functional as F
from .DesNet import densenet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
from .torch_linear.model_ import GACNN as GA_model_linear

class ZeroPad2d(nn.ConstantPad2d):
    # Pads the input tensor boundaries with zero.
    def __init__(self, padding):
        super(ZeroPad2d, self).__init__(padding, 0)


class crop_model_cat(nn.Module):
    def __init__(self, in_channel, num_classes, model = None, need_return_dict = False, middle_layer = 8) -> None:
        super().__init__()
        self.middle_layer_key = middle_layer
        middle_layer_number = (self.middle_layer_key+2) *  (self.middle_layer_key+2) 
        self.flatten = torch.nn.Flatten()
        self.line = torch.nn.Linear(middle_layer_number, 2)
        if model != None:
            self.model = model
        else:
            self.model = densenet(1, num_classes, need_return_dict = need_return_dict)
        self.need_return_dict = need_return_dict

        self.ga_model = GA_model_linear(1, num_classes)
        self.pad = ZeroPad2d(1)
        self.in_channel = in_channel

    def feature(self, x):
        x = self.ga_model(x)
        return x

    def build_ans(self,x, y):
        return {
            "map":x,
            "pred_logits":y,
        }

    def revert_tensor_col(self,image_pack, batch):
        ans = []
        for i in range(batch):
            ans.append(image_pack[i::batch,:])
        return torch.stack(ans)

    def revert_tensor_raw(self,image_pack, batch):
        return image_pack[:,0].reshape(self.middle_layer_key, self.middle_layer_key)
        #if(batch == 1):
            #image_pack[:,0]
            #return image_pack[:,0].reshape(self.middle_layer_key, self.middle_layer_key)
        # else:
        #     return image_pack[:,0].reshape(batch, self.in_channel, self.middle_layer_key, self.middle_layer_key)



    def forward(self, input, return_feature_map = False):

        p, c, _, _ = input.size()

        img_class = self.model(input)
        #print(a.size())
        feature_map = self.revert_tensor_raw(img_class, p)
        a = self.pad(feature_map)
        a = a.unsqueeze(0)
        
        #print(feature_map.size(), a.size())
        a = self.feature(a)
        #print(x.size())
        return self.build_ans(img_class, a) if self.need_return_dict else (img_class, a)

    def crop_tensor(self, image_pack, scale = 4):
        _, _, w, h = image_pack.size()
        a = int(w/scale)
        b = int(h/scale)
        t = torch.split(image_pack, a, dim = 2)
        ans = []
        for i in t:
            for j in torch.split(i,b, dim=3):
                ans.append(j)
        d = torch.cat(ans, 0)
        return d

class Complex_loss():
    def __init__(self):
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
    def __call__(self, logits, maps, gt_logits, gt_map):
        logits_loss = self.CrossEntropyLoss(logits, gt_logits)
        mse_loss = self.MSELoss(maps, gt_map)
        return logits_loss + mse_loss