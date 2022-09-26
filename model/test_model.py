import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
from cmt_pytorch.cmt import PoCMT, CMT,PoCMT_ti
from torch_linear.model_ import *
import GPUtil
use_gpu = torch.cuda.is_available()
if(use_gpu):
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.8, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
    print("detect set :",deviceIDs)
    device = torch.device("cuda:"+str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)
if __name__ == "__main__":
    

    batch_image = torch.rand((2,1,256,256))
    print(batch_image.size())
    cmt_model = PoCMT_ti(img_size=256, in_chans=1, num_classes=2, size_type = 2, need_return_dict = False )
    print(cmt_model)
    ans = cmt_model(batch_image)
    print(ans.size())
    print(ans)

