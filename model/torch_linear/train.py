from .data_generate_h5 import DataGenerate
from .model_ import CNN as CNN_Model
from .model_ import SMCNN as SMCNN_Model
from .model_ import GACNN as GACNN_Model
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
import GPUtil
use_gpu = torch.cuda.is_available()
if(use_gpu):
    deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.8, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
    
    if(len(deviceIDs) == 0):
        deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 1, maxMemory = 1, includeNan=False, excludeID=[], excludeUUID=[])
    print(deviceIDs)
    device = torch.device("cuda:"+str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)

class Train():
    def __init__(self, in_channles, out_channels, is_show = True, type = 0):
        print("GA:")
        print("using device:",device)
        print("use gpu:", use_gpu)
        self.in_channels = in_channles
        self.out_channels = out_channels
        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.type = type
        self.create(is_show)
    
    def create(self, is_show):
        print(__name__,"use:",device)
        if(self.type == 1):
            self.model = CNN_Model(self.in_channels, self.out_channels)
        elif(self.type == 2):
            self.model = GACNN_Model(self.in_channels, self.out_channels)
            print("build GACNN_Model")
        else:
            self.model = SMCNN_Model(self.in_channels, self.out_channels)
            
        self.cost = torch.nn.CrossEntropyLoss()
        if(use_gpu):
            self.model = self.model.to(device)
            self.cost = self.cost.to(device)
        if(is_show):
            self.model.summary()
        
        #self.cost = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.5, 0.999))
   
    def train(self, n_epochs, data_loader_train, data_loader_test):
        self.model.train()
        for epoch in range(n_epochs):
            running_loss = 0.0
            running_correct = 0
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-"*10)
            train_index = 0
            for data in data_loader_train:
                X_train, y_train = data
                X_train, y_train = Variable(X_train).float(), Variable(y_train)
                #print(X_train.type())
                self.optimizer.zero_grad()
                #print(X_train.size())
                outputs  = self.model(X_train)
                #return outputs,outputs_queko,outputs_pingko,y_train
                #_,pred = torch.max(outputs.data, 2)
                
                #print("output:", outputs)
                #print("y:", y_train)
                
                y_train = y_train.long()
                
                loss_liewen = self.cost(outputs, y_train.long())
              
                    
                loss = loss_liewen
           
                loss.backward()
                self.optimizer.step()
                #print(loss.data.item())
                running_loss += loss.data.item()
                
                _,pred = torch.max(outputs.data, 1)
               
                ans1 =  [ 1 if y_train[index] == pred[index] else 0 for index in range(pred.size()[0])]
                
                running_correct += np.sum(ans1) / len(ans1)
                train_index += 1
            
            
            self.model.eval()
            testing_correct = 0
            running_test_loss = 0
            test_index = 0
            with torch.no_grad():
                for data in data_loader_test:
                    X_test, y_test = data
                    X_test, y_test = Variable(X_test).float(), Variable(y_test)
                    outputs = self.model(X_test)
                    #loss = self.cost(outputs, y_test)
                    #loss = 0
                    loss = self.cost(outputs, y_test.long())
                    
                    running_loss += loss.data.item()

                    _,pred = torch.max(outputs.data, 1)

                    ans1 =  [ 1 if y_test[index] == pred[index] else 0 for index in range(pred.size()[0])]

                    
                    running_test_loss += loss.data.item()
                    
                    #ans =  [ 1 if y_test_flatten[index] == pred_flatten[index] else 0 for index in range(pred_flatten.size()[0])]
                    testing_correct += np.sum(ans1)  / len(ans1)
                    test_index += 1
            epoch_loss = running_loss/train_index
            #print( running_correct,  train_index)
            epoch_acc = 100*running_correct/train_index
            epoch_test_loss = running_test_loss/test_index
            epoch_test_acc = 100*testing_correct/test_index
            self.history_acc.append(epoch_loss)
            self.history_loss.append(epoch_acc)
            self.history_test_loss.append(epoch_test_loss)
            self.history_test_acc.append(epoch_test_acc)
            print(
                "Loss is:{:.4f}, Train Accuracy is:{:.4f}, Loss is{:.4f}, Test Accuracy is:{:.4f}".format(
                    epoch_loss,
                    epoch_acc,
                    epoch_test_loss,
                    epoch_test_acc,
                )
             )
        self.save_history()
        self.save_parameter()

    def predict(self, image):

        if(type(image) == np.ndarray):
            image = torch.from_numpy(image)
        if(len(image.size()) == 3 ):
            image.unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            image = Variable(image).float()
            if(use_gpu):
                image = image.to(device)
            output = self.model(image )
            _, preds = torch.max(output.data, 1)
        return preds
    
    def save_history(self, file_path = './save/'):
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "loss_test_history.txt", "w+")
        fo.write(str(self.history_test_loss))
        fo.close()   
        fo = open(file_path + "test_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close() 
    def save_parameter(self, file_path = './save/'):
        if not os.path.exists(file_path): 
            os.mkdir(file_path)
        file_path = file_path + "model_" +str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace("-","_").replace(".","_") + ".pkl"
        torch.save(obj=self.model.state_dict(), f=file_path)
    def load_parameter(self, file_path = './save/' ):
        # self.model.load_state_dict(torch.load('model_parameter.pkl'))
        if use_gpu:
            self.model.load_state_dict(torch.load(file_path,map_location=device))
        else:
            self.model.load_state_dict(torch.load(file_path, map_location='cpu'))
    
if __name__ == "__main__":
    batch_size = 32
    train_dir_path = "F:/outpage/Virsualiz-torch-liear/data/" 
    dg = DataGenerate(train_dir_path, batch_size = batch_size)
    dg.split_train_and_test()
    trainer = Train(1, 2)
    train_dataloader = []
    test_dataloader = []
    for i in dg.train_iter()():
        train_dataloader.append(i)
    for i in dg.test_iter()():
        test_dataloader.append(i)
    print(len(train_dataloader), len(test_dataloader))
    trainer.train(100, train_dataloader, test_dataloader)
