from cProfile import label
from functools import total_ordering
from operator import le
from turtle import left, width
from matplotlib import image
import numpy as np
import h5py
import os
import math
import glob
import datetime
#from lib.util import *
import copy
import torch
from torchvision import transforms as transforms


#TYPE_TEST = "DEBUG"
#TYPE_TEST = "false"


class DataGenerate():
    def __init__(self, train_dir_path = './data/', transform=None , batch_size = 16):
        self.checkdir(train_dir_path)

        self.train_dir_path = train_dir_path
        self.train_file_path_list = self.get_file_path_list(self.train_dir_path, '*.h5')
        
        self.out_path = None
        self.batch_size = batch_size
        self.label_list = {}
        
       
        self.source_file = []
        self.source_file_len = {
            "yes":0,
            "no":0,
        }
        self.transform = transform
        self.data_index = []
        self.train_index = []
        self.test_index = []
        self.val_index = []
        for path in self.train_file_path_list:
            self.load_h5(path)
        #self.crop_map(data_set_number = "L", files = self.train_file_path_list)
    
    def makedir(self, dir_path):
        if(not os.path.exists(dir_path)):
            os.mkdir(dir_path)
    
    def checkdir(self, dir_path):
        if(os.path.exists(dir_path)):
            return True
        else:
            assert False, "dir"+ dir_path + "not exists!!!"
    
    def get_file_path_list(self, dir_path, file_type = ".fits"):
        if(dir_path == None):
            return None
        files=glob.glob(dir_path+file_type)
        return files
    
    def get_data(self, data_index, need_detail = True): 
        
        data = []
        label = []
        for d_index in data_index:
            data.append(self.data[d_index])
            label.append(self.label[d_index])
        data = np.array(data, dtype= np.uint8)  
        label = np.array(label)
        
        data = np.expand_dims(data, 1)
      
       
        if(self.transform is None):
            return torch.from_numpy(data), torch.from_numpy(label), 
        else:
            return self.transform(data), self.transform(label), 

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        pass

    def train_iter(self, drop_last = True):
        def callback():
            total = int( len(self.train_index)/self.batch_size ) if drop_last else math.ceil(len(self.train_index)/self.batch_size)
            # total = int(self.data_index["train_index_end"] / self.batch_size) if drop_last else math.ceil(self.data_index["train_index_end"] / self.batch_size)
            # print(total, self.data_index["train_index_end"] / self.batch_size)
            for batch in range(total):
                yield self.get_data(self.train_index[batch*self.batch_size:(batch+1)*self.batch_size])
                #return 
        return callback

    def test_iter(self, drop_last = True):
        def callback():
            total = int( len(self.test_index)/self.batch_size ) if drop_last else math.ceil(len(self.test_index)/self.batch_size)
            # iter_total = int( total / self.batch_size) if drop_last else math.ceil(total / self.batch_size)
            #p rint(total, iter_total)
            for batch in range(total):
                yield self.get_data( self.test_index[batch*self.batch_size:(batch+1)*self.batch_size])
                #return
        return callback

    def val_iter(self, drop_last = True):
        def callback():
            total = int( len(self.val_index)/self.batch_size ) if drop_last else math.ceil(len(self.val_index)/self.batch_size)
            # iter_total = int( total / self.batch_size) if drop_last else math.ceil(total / self.batch_size)
            for batch in range(total):
                yield self.get_data(self.val_index[batch*self.batch_size:(batch+1)*self.batch_size]) 
        return callback

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return self
       
    def __del__(self):
        print("释放内存")
        for hdffile in self.source_file:
            hdffile.close()
            print(hdffile)

    def printname(self, name):
        print(name)

    def split_train_and_test(self, shuffle = True, train_size = 0.8):
        total = self.source_file_len["yes"] + self.source_file_len["no"]
        file_index = np.arange(0, total, dtype = np.int)
        if(shuffle):
            np.random.shuffle(file_index)
        train_end = int( train_size * total)
        temp_index = int( (total - train_end)/2 )
        val_end = train_end + temp_index

        self.train_index = file_index[:train_end]
        self.val_index = file_index[train_end:val_end]
        self.test_index =  file_index[val_end:]

        print("train size:{}, val size:{}, test size{}".format(
            len(self.train_index),

            len(self.val_index),

            len(self.test_index),
        ))

    #@log(TYPE_TEST, "load h5")                
    def load_h5(self, files_path):
        hf = h5py.File(files_path, "r")
        print("load path:",files_path,"\n")
        #logging.info("load path:"+files_path)
        #print("dict value:")
        #hf.visit(self.printname)
        self.source_file.append(hf)
        self.data = []
        yes = len(hf["data_yes"])
        no = len(hf["data_no"])
        self.data = np.array(hf["data_yes"][:])
        self.label = np.ones(len(hf["data_yes"]))
        self.data = np.concatenate( [self.data,  np.array(hf["data_no"][:])],0)
        self.label = np.concatenate([self.label,  np.zeros(len(hf["data_no"]))],0)

        self.source_file_len["yes"] = yes
        self.source_file_len["no"] = no
    
    def encode_T(self, map, crop_image_list):
        w,h = map.shape
        ans_vector = []
        for i in crop_image_list:
            row_256_s, row_256_e, col_256_s, col_256_e = i["coord"]
            row_256_s = row_256_s%2048
            down = 0
            top = 0
            right = 0
            left = 0
            
            left_top = 0
            right_top = 0
            left_down = 0
            right_down = 0
            x,y = int(row_256_s/256) + 1 , int(col_256_s/256)+ 1
            
            down = map[ x + 1, y]
            
            top = map[ x - 1, y]
            
            
            left = map[ x, y - 1 ]
            right = map[ x, y + 1 ]
            
            right_top = map[ x - 1, y + 1 ]
            right_down = map[ x + 1, y + 1 ]
            left_top = map[ x - 1, y - 1 ]
            left_down = map[ x + 1, y - 1 ]
            
            if(down + top + right + left + right_top + right_down + left_top + left_down == 0): #孤立点
                ans_vector.append(0)
            elif(down + top + right + left == 2):
                ans_vector.append(1)
            else:
                ans_vector.append(1)

        return ans_vector
            
    def encode_F(self, map, crop_image_list):
        w,h = map.shape
        ans_vector = []
        for i in crop_image_list:
            row_256_s, row_256_e, col_256_s, col_256_e = i["coord"]
            row_256_s = row_256_s%2048
            down = 0
            top = 0
            right = 0
            left = 0
            if( int(row_256_s/256) -1 >= 0 and int(row_256_s/256) -1 <= h):
                down = map[int(row_256_s/256) -1 , int(col_256_s/256)]
            if( int(row_256_s/256) + 1 >= 0 and int(row_256_s/256) + 1 <= h):
                top = map[int(row_256_s/256) + 1 , int(col_256_s/256)]
            if( int(col_256_s/256)+ 1 >= 0 and int(col_256_s/256)+ 1 <= w):
                right = map[int(row_256_s/256), int(col_256_s/256)+ 1]
            if( int(col_256_s/256)- 1 >= 0 and int(col_256_s/256)- 1 <= w):
                left = map[int(row_256_s/256), int(col_256_s/256)- 1]
            if(down + top == 2 or  right + left == 2):
                ans_vector.append(0)
            else:
                ans_vector.append(1)
        return ans_vector

    def coding_absence(self, data , show = True, vector = None, label = 0):
        image = np.zeros((10,10))
        #     image[:,0] = 1
        #     image[0,:] = 1
        #     image[-1,:] =1
        #     image[:,-1] = 1
        crop_image_list = data["256"]
        total = len(crop_image_list)

        for index in range(total):
            i = crop_image_list[index]
            row_256_s, row_256_e, col_256_s, col_256_e = i["coord"]
            row_256_s = row_256_s%2048
            #col_256_s = col_256_s/256
            if(vector is not None):
                image[ int(row_256_s/256) +1, int(col_256_s/256) +1] = vector[index]
            else:
                if(label):
                    image[ int(row_256_s/256) +1, int(col_256_s/256) +1] = i["label"]
                else:
                    image[ int(row_256_s/256) +1, int(col_256_s/256) +1] = 0 if i["label"] else 1

        if(show):
            plt.imshow( image )
            plt.show()
        return image
            
    def sort_data(self, data_dict):
        data = []
        for i in data_dict["2048"]:
            row_2048_s, row_2048_e, col_2048_s, col_2048_e = i["coord"]
            temp = {
                "2048":i,
                "256":[],
            }
            for j in data_dict["256"]:
                row_256_s, row_256_e, col_256_s, col_256_e = j["coord"]
                if(row_2048_s <= row_256_s and row_256_e <= row_2048_e and col_2048_s <= col_256_s and col_256_e <= col_2048_e):
                    temp["256"].append(j)
            data.append(temp)
        return data

    def get_group_by_index(self, file_index, group_index):
        file_ = self.source_file[file_index]
        #print(file_)
        data_dict = {"256":[],"2048":[]}
        total = len(file_["256"]["image"])
        for index in range(total):
            source_index = file_["256"]["index"][index]
            if(source_index == group_index):
                i = {
                    "image":file_["256"]["image"][index],
                    "coord":file_["256"]["coord"][index],
                    "label":file_["256"]["label"][index],
                    "index":source_index,
                }
                data_dict["256"].append(i)

        total = len(file_["2048"]["image"])
        for index in range(total):
            source_index = file_["2048"]["index"][index]
            if(source_index == group_index):
                i = {
                    "image":file_["2048"]["image"][index],
                    "coord":file_["2048"]["coord"][index],
                    "label":file_["2048"]["label"][index],
                    "index":source_index,
                }
                data_dict["2048"].append(i)
        return data_dict

    def save_data(self, save_name, data, save = True):
        dpi = 256
        height, width = data.shape
        plt.figure(figsize=(1.2993*height/dpi, 1.2993*width/dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(data)
        if(save):
            plt.savefig(save_name, bbox_inches='tight',pad_inches=0.0,dpi = dpi)
        plt.clf()

    def save_as_h5(self, file_path, name):
        import h5py
        f = h5py.File(file_path + name + "0.h5", "w")
        print(file_path+ name + "0.h5")
        image_list = {
            "yes":[],
            "no":[],
        }
        
        for i in range(len(self.source_file)):
            fits_index = self.source_file[i]["2048"]["index"][0]
            data_dict = self.get_group_by_index(i, fits_index)
            test_data = self.sort_data(data_dict)
            index = 1
            for data_group in test_data:
                label = data_group["2048"]["label"]
                if len(data_group["256"]) :
                    image = self.coding_absence(data_group, show = False , label = label)
                    if(label):
                        image_list["yes"].append(image)
                    else:
                        image_list["no"].append(image)
                index += 1

        print(len(image_list["yes"]), len(image_list["no"]))
        #g = f.create_group("data")
        f.create_dataset("data_yes", data=copy.deepcopy(image_list["yes"]))
        f.create_dataset("data_no", data=copy.deepcopy(image_list["no"]))
        f.close()

    def pipline(self, file_id = 0, number_of_file = 10, save_path_ = None):
        fits_index = self.source_file[file_id]["2048"]["index"][0]
        data_dict = self.get_group_by_index(file_id, fits_index)
        test_data = self.sort_data(data_dict)
        index = 1
        for data_group in test_data:
            label = data_group["2048"]["label"]
            if(label):
                save_path = save_path_ + "yes/" + str(file_id)+"/"
            else:
                save_path = save_path_ + "no/" + str(file_id)+"/"
            if len(data_group["256"]) :
                self.makedir(save_path)
                image = self.coding_absence(data_group, show = False , label = label)
                name = save_path + "_" + str(index)+ "_" +str(data_group["2048"]["label"]) +".png"
                self.save_data(name, image)
            index += 1
            


if __name__ == "__main__":
    
    batch_size = 32
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # train_dir_path = "H:/DATASET/FRBDATA/GTTWTDATA/gttwt14/insert_frb/"
    #start_time = datetime.datetime.now()
    #dg = DataGenerate(train_dir_path = train_dir_path, data_set_number = "S",transform = None, batch_size = 12)
    # dg.split_train_and_test()
    # end_time = datetime.datetime.now()
    # print("cost time:",(end_time - start_time).seconds,"seconds")

    # train_i = dg.train_iter()
    # for i in train_i:
    #     image_list = i[0]
    #     image_label = i[1]
    #     break

    # print(image_list.shape,image_label.shape)
    # plt.imshow(image_list[-1])
    # plt.title(str(image_label[-1][1]))
    # plt.show()

    # del dg
    train_dir_path = "I:\\19_C1_h5/"
    dg = DataGenerate(train_dir_path = train_dir_path, data_set_number = "S",transform = None, batch_size = 12)
    out_path = "H:/DATASET/FRBDATA/test/"
    dg.save_as_h5(out_path, "sa")

    

