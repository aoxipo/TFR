#from model.resnet_model import ResNet17 as Model
#from model.InceptionresNetV2 import inceptionresnetv2 as Model
#from model.DesNet import densenet as Model
#from model.efficientnet_pytorch.model import EfficientNet
#from model.efficientnet_pytorch.utils import get_blocks_args_global_params_b4
#from model.detr_.models.detr import MYDETR as Model
#from model.detr_.models.transformer import build_easy_transformer
from data_generate_h5 import DataGenerate
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
import cv2
import copy
import matplotlib.pyplot as plt
from model.data_warpper import DaTa_Warpper
from torchvision import transforms as transforms

#from .model.cmt_pytorch.cmt import PoCMT as Model
from lib.util import *
import GPUtil
# set random seed
# torch.manual_seed(3407)
use_gpu = torch.cuda.is_available()
if(use_gpu):
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.8, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
    if(len(deviceIDs) == 0):
        deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 1, maxMemory = 1, includeNan=False, excludeID=[], excludeUUID=[])
    print("detect set :",deviceIDs)
    device = torch.device("cuda:"+str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)

class Train():
    def __init__(self, image_shape = (256, 256), class_number = 2, is_show = False, name = "None", method_type = 0):

        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.class_number = class_number
        self.image_shape = image_shape
        self.name = name
        self.type = method_type
        self.data_warpper = None
        self.create(is_show)

    def create(self, is_show):
        """
        trainsformer = build_easy_transformer(
            hidden_dim  = 128,
            dropout = 0.1,
            nheads = 2,
            dim_feedforward = 512 ,
            enc_layers = 3,
            dec_layers = 3,
            pre_norm = False,
        )

        self.model = Model( trainsformer, 2, 1)
        """
        self.data_warpper = DaTa_Warpper()
        if self.type == 0:
            from model.resnet_model import ResNet17 as Model
            self.model = Model(need_return_dict = True)
            print("ResNet17")
        elif self.type == 1:
            from model.InceptionresNetV2 import inceptionresnetv2 as Model
            self.model = Model(num_classes=2, pretrained=None,need_return_dict = True)
            print("inceptionresnetv2")
        elif self.type == 2:
            from model.DesNet import densenet as Model
            self.model = Model(in_channel=1, num_classes=2,need_return_dict = True)
            print("dense121")
        elif self.type == 3:
            from model.efficientnet_pytorch.model import EfficientNet as Model
            from model.efficientnet_pytorch.utils import get_blocks_args_global_params_b6
            a,b = get_blocks_args_global_params_b6()
            self.model = Model(a,b,need_return_dict = True)
            print("EfficientNet-b06")
        # elif self.type == 34:
        #     from model.efficientnet_pytorch.model import EfficientNet as Model
        #     from model.efficientnet_pytorch.utils import get_blocks_args_global_params_b4
        #     a,b = get_blocks_args_global_params_b4()
        #     self.model = Model(a,b)
        #     print("EfficientNet-b04")
        elif self.type == 4:
            from model.cmt_pytorch.cmt import PoCMT_xs as Model #PoCMT_ti PoCMT_xs PoCMT_b PoCMT_s
            self.model = Model(img_size=256, in_chans=1, num_classes=2, need_return_dict = True, size_type = 2 )
            print("CMT model")
        else:
            raise NotImplementedError
        #self.model = Model(num_classes=2, pretrained=None)
        #self.model = Model(in_channel=1, num_classes=2)
        if(is_show):
            summary(self.model, (1,self.image_shape[0], self.image_shape[1]))
        self.cost_class = torch.nn.CrossEntropyLoss()
   
        if(use_gpu):
            self.model = self.model.to(device)
            self.cost_class = self.cost_class.to(device)#cuda()
            self.data_warpper = self.data_warpper.to(device)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0.001, )
     
    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        logging.debug("start train and test")
        
        best_acc = -2
        for epoch in range(n_epochs):
            start_time = datetime.datetime.now()
           
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-"*10)
            
            
            train_loss, train_correct = self.train(data_loader_train) 
            testing_loss, testing_correct = self.test(data_loader_test)
            
            epoch_loss = train_loss
            epoch_acc = train_correct
            epoch_test_loss = testing_loss
            epoch_test_acc = testing_correct
            
            self.history_acc.append(epoch_acc)
            self.history_loss.append(epoch_loss)
            self.history_test_acc.append(epoch_test_acc)
            self.history_test_loss.append(epoch_test_loss)
            print(
                "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is:{:.4f}, Test Accuracy is:{:.4f} ,cost time:{:.4f} min, EAT:{:.4f} min".format(
                    epoch_loss,
                    epoch_acc,
                    epoch_test_loss,
                    epoch_test_acc,
                    (datetime.datetime.now() - start_time).seconds/60, 
                    (n_epochs - 1 - epoch)*(datetime.datetime.now() - start_time).seconds/60,
                )
             )
            
            if ((epoch_test_acc > 95 and epoch_test_acc > best_acc) or  (epoch_test_acc <= 95 and epoch_test_acc > best_acc)):# + 1) ) :
                best_acc = epoch_test_acc
                es = 0
                self.save_parameter("./save_best/", "best")
            else:
                es += 1
                print("Counter {} of 10".format(es))

                if es > 4:
                    print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", epoch_test_acc, "...")
                    break
            logging.debug(
                "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Loss is{:.4f}, Test Accuracy is:{:.4f} ,cost time:{:.4f} min, EAT:{:.4f} min".format(
                    epoch_loss,
                    epoch_acc,
                    epoch_test_loss,
                    epoch_test_acc,
                    (datetime.datetime.now() - start_time).seconds/60, 
                    (n_epochs - 1 - epoch)*(datetime.datetime.now() - start_time).seconds/60,
                )
             )
            
        self.save_parameter()
        self.save_history()
            
                
    def train(self, data_loader_train):
        #logging.debug("start train")
        self.model.train()  
        running_loss = 0.0
        running_correct = 0
        train_index = 0
        for data in data_loader_train():
            X_train, x_class  = data
            #print(X_train.dtype)
            X_train, x_class = Variable(X_train).type(torch.float32), Variable(x_class)
            if(use_gpu):
                X_train = X_train.to(device)
                x_class = x_class.to(device)
                    
            self.optimizer.zero_grad()
            #print(X_train.type())
            if(self.data_warpper is not None):
                X_train = self.data_warpper(X_train)
            outputs = self.model(X_train)
            #print(outputs['pred_logits'].size())
            #outputs['pred_logits'] = outputs['pred_logits'].squeeze(0)
            #print(outputs['pred_logits'].size())
            _,pred = torch.max(outputs['pred_logits'], 1)
            _,pred_label = torch.max(x_class, 1)
            #                 y_train = y_train.long()

            #x_class = x_class.float()
            #print(outputs['pred_logits'].size())
            loss = self.cost_class(outputs['pred_logits'], pred_label)
            
            #return 0,0
            loss.backward()
            self.optimizer.step()
                
            acc = [ 1 if pred_label[i] == pred[i] else 0 for i in range(len(pred)) ]
            #print("train:",outputs['pred_logits'],acc)
            #print(X_train[0])
            if(use_gpu):
                running_loss += loss.cpu().data.item()
                running_correct += sum(acc)/len(pred)
            else:
                running_loss += loss.data.item()
                running_correct += sum(acc)/len(pred)
              
            train_index += 1
               
        print("total train:",train_index)
            
        return running_loss/train_index , running_correct/train_index * 100
            
    
    def test(self, data_loader_test):

        self.model.eval()
        with torch.no_grad():
            testing_loss = 0
            testing_correct = 0
            test_index = 1
            for data in data_loader_test():
                # print("iter {}/{}".format(test_index, total))
                X_train, x_class  = data
               
                X_train, x_class = Variable(X_train).type(torch.float32), Variable(x_class)
                if(use_gpu):
                    X_train = X_train.to(device)
                 
                    x_class = x_class.to(device)

                if(self.data_warpper is not None):
                    X_train = self.data_warpper(X_train)
                outputs = self.model(X_train)
                
                _,pred = torch.max(outputs['pred_logits'], 1)
                _,pred_label = torch.max(x_class, 1)

                #x_class = x_class.float()
                
                loss = self.cost_class(outputs['pred_logits'], pred_label)
                
                acc = [ 1 if pred_label[i] == pred[i] else 0 for i in range(len(pred)) ]
                  
                #print("test:",outputs['pred_logits'], acc)
                #print(X_train[0])
                if(use_gpu):
                    testing_loss += loss.cpu().data.item()
                    testing_correct += sum(acc)/len(pred)
                else:
                    testing_loss += loss.data.item()
                    testing_correct += sum(acc)/len(pred)

                test_index += 1
            print("total test:",test_index)
            epoch_test_loss = testing_loss/test_index
            epoch_test_acc = testing_correct/test_index * 100
                
        return  epoch_test_loss, epoch_test_acc
        #self.save_history(save_mode= "test" )
    
    # fits文件 单张预测 无label   
    def predict_signle(self,data_loader_val):
        self.model.eval()
        with torch.no_grad():
            pred_ans = []
            for data_group in data_loader_val["2048"]:
                #print(data_group.shape)
                test_256_image = self.crop_tensor( torch.from_numpy(data_group), 8)
                if(self.data_warpper is not None):
                    test_256_image = self.data_warpper(test_256_image)
                #print(test_256_image.shape)
                X_test = test_256_image.unsqueeze(1)
                #print(X_test.shape)
                X_test = Variable(X_test).type(torch.float32)

                if(use_gpu):
                    X_test = X_test.to(device)
                outputs = self.model(X_test)
                _,pred = torch.max(outputs['pred_logits'], 1)
                pred_ans.append(pred.cpu().numpy())
        return pred_ans
    
    def crop_tensor(self, image_pack, scale = 4):
        if len(image_pack.size()) == 2:
            image_pack = image_pack.unsqueeze(0)
            _, w, h = image_pack.size()
            dim1 = 1
            dim2 = 2
            cat_dim = 0
        else:
            _, _, w, h = image_pack.size()
            dim1 = 2
            dim2 = 3
            cat_dim = 1

        a = int(w/scale)
        b = int(h/scale)
        t = torch.split(image_pack, a, dim = dim1)
        ans = []
        for i in t:
            for j in torch.split(i,b, dim = dim2):
                ans.append(j)
        d = torch.cat(ans, cat_dim)
        return d
    
    # 
    def predict(self, data_loader_val, need_detail = False):
        vector_list = []
        self.model.eval()  
        with torch.no_grad():
            testing_loss = 0
            testing_correct = 0
            test_index = 1
            #total = len(data_loader_val["2048"]) 
            ans_label_list = []
            for data_group in data_loader_val["2048"]: 
                #print(data_group["image"])
                # print("iter {}/{}".format(test_index, total))
                label = data_group["label"]
                
                test_256_image = self.crop_tensor( torch.from_numpy(data_group["image"]), 8)
                #print(test_256_image.size(), test_256_image.dtype)
                if(self.data_warpper is not None):
                    test_256_image = self.data_warpper(test_256_image)
                #print(test_256_image.size())
                #test_256_image = resize(test_256_image)
                #print(test_256_image.size())
                # offset = torch.mean(test_256_image, 1)
                # for index in range(len(test_256_image)):
                #     test_256_image[index] = test_256_image[index] - offset[index]
                
                test_256_label = data_group["map"].reshape(-1)
                
                X_test = test_256_image
                X_test = X_test.unsqueeze(1)
                X_test = Variable(X_test).type(torch.float32)
                #print(X_test.size())

                if(use_gpu):
                    X_test = X_test.to(device)
                outputs = self.model(X_test)
            
                _,pred = torch.max(outputs["pred_logits"], 1)
                ans_label_list.append([pred, test_256_label])
                #print(pred)
                #print(":",test_256_label)
                #print(outputs['pred_logits'])
                #print(pred,pred.shape)
                #print(test_256_label.shape)
                
                acc = [ test_256_label[i] == pred[i] for i in range(len(pred)) ]

                testing_correct += sum(acc)/len(pred)
               
                test_index += 1
                vector_list.append([  1 if pred[i] else 0 for i in range(len(pred)) ])
            epoch_test_acc = testing_correct/test_index
            #print(epoch_test_acc)
        if(need_detail):
            return epoch_test_acc, np.array(vector_list), ans_label_list
        else:
            return  epoch_test_acc, np.array(vector_list)
        #self.save_history(save_mode= "test" )
    
    # 4096 -> resize 256 预测
    def predict_2048(self, data_loader_val, need_detail = False):
        vector_list = []
        self.model.eval()
        with torch.no_grad():
            testing_loss = 0
            testing_correct = 0
            test_index = 1
            total = len(data_loader_val) 
            ans_label_list = []
            for data_group in data_loader_val: 
                # print("iter {}/{}".format(test_index, total))
                label = [data_group["2048"]["label"]]
                image = np.array( data_group["2048"]["image"] )
                image = cv2.resize(image,(256,256))
                #print(image.shape)
                X_test = torch.from_numpy( image )
                X_test = X_test.unsqueeze(0)
                X_test = X_test.unsqueeze(0)
                X_test = Variable(X_test).type(torch.float32)
                #print(X_test.size())
                if(use_gpu):
                    X_test = X_test.to(device)
                outputs = self.model(X_test)

                _,pred = torch.max(outputs['pred_logits'], 1)
                ans_label_list.append([pred.cpu(), label])
                #print(pred)
                acc = [ label[i] == pred[i] for i in range(len(pred)) ]

                testing_correct += sum(acc)/len(pred)
               
                test_index += 1
                vector_list.append([  1 if pred[i] else 0 for i in range(len(pred)) ])
            epoch_test_acc = testing_correct/test_index
        if(need_detail):
            return epoch_test_acc, vector_list, ans_label_list
        else:
            return  epoch_test_acc, vector_list
        #self.save_history(save_mode= "test" )
    
    def save_history(self, file_path = './save/'):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "test_acc_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close()   
        fo = open(file_path + "test_loss_history.txt", "w+")
        fo.write(str(self.history_test_loss))
        fo.close()   
        
    def save_parameter(self, file_path = './save/', name = None):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        if name is None:
            file_path = file_path + "model_" +str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace("-","_").replace(".","_") + ".pkl"
        else:
            file_path = file_path + name + ".pkl"
        torch.save(obj=self.model.state_dict(), f=file_path)

    def load_parameter(self, file_path = './save/' ):
        # self.model.load_state_dict(torch.load('model_parameter.pkl'))
        if use_gpu:
            self.model.load_state_dict(torch.load(file_path))
        else:
            self.model.load_state_dict(torch.load(file_path, map_location='cpu'))

if __name__ == "__main__":
    

    batch_size = 1
    #train_dir_path = r"E:\ljl\Dataset\Frbdata\MixData/"
    #train_dir_path = r"E:\ljl\Dataset\Frbdata\DataSet/"
    #train_dir_path = r"/home/data/lijl/DATA/Frbdata/SmallTrain/"
    #train_dir_path = r"/home/data/lijl/DATA/Frbdata/Wang/front/"
    #train_dir_path = r"/home/data/lijl/DATA/Frbdata/Wang/back/"
    #train_dir_path = r"/home/data/lijl/DATA/Frbdata/Wang/combine/"
    train_dir_path = r"/home/data/lijl/DATA/Frbdata/Wang/hard/"
    #train_dir_path = r"/home/data/lijl/DATA/Frbdata/Wang/OTSU_all/"
    #train_dir_path = "I:/19_C1_h5/"
    start_time = datetime.datetime.now()
    data_shape = (4096, 4096)
    dg = DataGenerate(train_dir_path = train_dir_path, data_set_number = "S", batch_size = batch_size, data_shape = data_shape)
    dg.split_train_and_test(train_size = 0.8, keep_same = False)
    dg.normal = True
    device = 'cuda:1'
    print("using device:",device)
    print("use gpu:", use_gpu)
    print("using normal:",dg.normal)
    end_time = datetime.datetime.now()
    print("cost time:",(end_time - start_time).seconds,"seconds")
    dg.data_key = "2048"
    print("use key:",dg.data_key)
    logging.debug("cost time:{}".format((end_time - start_time).seconds,"seconds"))
    #logging.debug("use key:{}"%dg.data_key)
    print("using batch:", batch_size)

    method_dict ={
        "conv17":0,
        "inceptionresnetv2":1,
        "dense121":2,
        "efficientnet":3,
        "cmt":4,
    }

    trainer = Train(
        image_shape = data_shape,
        class_number = 2, 
        is_show = False,
        name = "efficientnet_8x8_2048_transfer_hard_all_4",
        method_type = 3
    )
    #print("load parameters")
    #trainer.load_parameter(r"E:\ljl\conv17\save_best\efficientnet_8x8_2048/best.pkl")
    trainer.load_parameter(r"/home/data/lijl/PROJECT/conv17_clean/save_best/efficientnet_8x8_2048/best.pkl")
    #trainer.train_and_test(100, dg.train_iter(), dg.test_iter())
    trainer.train_and_test(100, dg.train_iter(batch_size = 32), dg.test_iter(batch_size = 32))
    #trainer.test(dg.test_iter())
