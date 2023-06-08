import numpy as np
import h5py
import os
import math
import glob
import datetime

import matplotlib
matplotlib.use('AGG') # Fail to allocate bitmap
import matplotlib.pyplot as plt

from lib.util import *
import copy
import torch
import cv2
from torchvision import transforms as transforms
import astropy.io.fits as pyfits
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


#TYPE_TEST = "DEBUG"
#TYPE_TEST = "false"

SCALE = 8
class DataGenerate():
    def __init__(self, train_dir_path = './data/', out_path = "./save/", data_set_number = "L", transform=None , batch_size = 16, file_type = '*.h5', data_shape = (256,256)):
        self.checkdir(train_dir_path)
        self.normal = True
        self.train_dir_path = train_dir_path
        self.train_file_path_list = self.get_file_path_list(self.train_dir_path, file_type = file_type)
        self.data_shape = data_shape
        self.out_path = None
        self.batch_size = batch_size
        self.label_list = {}
        self.data_set_scale = {
            "S": 256,
            "L": 2048,
            "LL":4096,
        }
        value = str(self.data_set_scale[data_set_number] )
        self.data_key = value
        self.source_file = []
        self.source_file_len = {
            "2048":[0],
            "2048_len":0,
            "256":[0],
            "256_len":0,
        }
        self.transform = transform
        self.data_index = {}
        self.train_index = {
            "256":[],
            "2048":[],
        }
        self.test_index = {
            "256":[],
            "2048":[],
        }
        self.val_index = {
            "256":[],
            "2048":[],
        }
        self.offset = []
        self.total_yes_train_end = {
            "256":0,
            "2048":0,
            }
        self.total_no_train_end = {
            "256":0,
            "2048":0,
            }
        #self.downsample = torch.nn.AvgPool2d(2,2)
        if(file_type == "*.h5"):
            for path in self.train_file_path_list:
                self.load_h5(path)
        #self.crop_map(data_set_number = "L", files = self.train_file_path_list)
    
        # def weight_normal(self, data, arrange = [[0,500, 0.5],[500, 10000, 0.4],[10000,-1, 0.1]] ):
        #     for phase in arrange:
        #         start = phase[0]
        #         end = phase[1]
        #         weight = phase[2]
        #         data[ data >= start and data <= end]/500 

    def makedir(self, dir_path):
        if(not os.path.exists(dir_path)):
            os.mkdir(dir_path)
    
    def checkdir(self, dir_path):
        if(os.path.exists(dir_path)):
            return True
        else:
            assert False, "dir "+ dir_path + "not exists!!!"
    
    def get_file_path_list(self, dir_path, file_type = ".fits"):
        if(dir_path == None):
            return None
        files=glob.glob(dir_path+file_type)
        files.sort()
        return files
    
    def get_data(self, data_index, need_detail = False): 
        label_vector = []
        coord_vector = []
        file_index_vector = []
        map_vector = []
        data = []
        try:
            for d_index in data_index:
                last_file_len = 0
                file_index = 0
                for i in self.source_file_len[self.data_key][1:]:
                    if(d_index < i):
                        #print(d_index - last, d_index)
                        image_index = int(d_index - last_file_len)
                        #print(image_index, int(d_index), int(last_file_len), int(d_index) - int(last_file_len))
                        image = self.source_file[file_index][self.data_key]["image"][image_index]
                        image_coord = self.source_file[file_index][self.data_key]["coord"][image_index]
                        
                        #print(image_coord)
                        #print(image.shape)
                        #if image.shape != self.data_shape:
                        #    downsample = torch.nn.AvgPool2d(2,2)
                        #    image = downsample(image)
                            
                        #image = image.astype(np.float32) 
                        #print(image.shape)
                        #print(self.data_shape)
                        if(self.normal):
                            image = (image - np.mean(image,0))#/(np.std(image, 0))
                            #image = (image - np.mean(image))/np.std(image)
                        
                        data.append(image)
                        label = self.source_file[file_index][self.data_key]["label"][image_index]
                        label_vector.append([0,1] if label else [1,0])
                        if(self.data_key == "2048"):
                            map_vector.append(self.source_file[file_index][self.data_key]["map"][image_index])
                        if(need_detail):
                            coord_vector.append( self.source_file[file_index][self.data_key]["coord"][image_index] )
                            file_index_vector.append( self.source_file[file_index][self.data_key]["index"][image_index] )
                        break
                    file_index += 1
                    last_file_len = i
        except Exception as e:
            print(e)
            print("d_index:",d_index, "last_file_len:",last_file_len,"file_index:", file_index,"i:", i,"image_index:", image_index)
            print(self.source_file_len[self.data_key][1:])
            raise RuntimeError("index error")
        
        data = np.array(data, dtype= np.float32)  
        label_vector = np.array(label_vector) 
        data = np.expand_dims(data, 1)
        
        data = torch.from_numpy(data)
        label_vector = torch.from_numpy(label_vector)
        
        
        if(need_detail):
            return data, label_vector, coord_vector, file_index_vector, 
        if(self.transform is None):
            if(self.data_key == "2048"):
                map_vector = torch.from_numpy(np.array(map_vector))
                return data, label_vector, map_vector
            else:
                return data, label_vector, 
        else:
            return self.transform(data), self.transform(label_vector), 

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        pass

    def crop_tensor(self, image_pack, scale = 4):
        
        if len(image_pack.size()) == 2:
            image_pack = image_pack.unsqueeze(0)
            image_pack = image_pack.unsqueeze(0)
        elif  len(image_pack.size()) == 3:
            image_pack = image_pack.unsqueeze(0)

        _, _, w, h = image_pack.size()
        dim1 = 2
        dim2 = 3
        cat_dim = 0

        a = int(w/scale)
        b = int(h/scale)
        t = torch.split(image_pack, a, dim = dim1)
        ans = []
        for i in t:
            for j in torch.split(i,b, dim = dim2):
                ans.append(j)
        d = torch.cat(ans, cat_dim)
        return d

    def get_8x8_from_2048(self, batch_size, data_index, drop_last = True):
        One_batch = SCALE * SCALE
        big_batch = int( (batch_size * One_batch)/math.gcd(batch_size, One_batch)/One_batch)
        print("big batch:",big_batch)
        def callback():
            total = int( len(data_index[self.data_key])/big_batch ) if drop_last else math.ceil(len(data_index[self.data_key])/big_batch)
            for batch in range(total):
                data, label, map_vector = self.get_data(data_index[self.data_key][batch*big_batch:(batch+1)*big_batch])
                #print(data.size(), map_vector.size())
                ans = []
                label_ans = []
                
                for i in range(big_batch):
                    ans.append(self.crop_tensor(data[i,:], SCALE))
                    map_label_list = map_vector[i,:].reshape(-1)
                   
                    label_ans.append(torch.tensor( [  [0,1] if label else [1,0] for label in map_label_list ]))
                
                crop_image = torch.cat(ans, 0)
                crop_label = torch.cat(label_ans, 0)
                #print(crop_image.size(), crop_label.size())
                crop_total = int(big_batch * One_batch/batch_size)
                #print(crop_total)
                for crop_batch in range(crop_total):
                    yield crop_image[ crop_batch*batch_size:(crop_batch+1) * batch_size ], crop_label[ crop_batch*batch_size:(crop_batch+1) * batch_size ]
        return callback

    def train_iter(self, drop_last = True, batch_size = -1):
        def callback():
            total = int( len(self.train_index[self.data_key])/self.batch_size ) if drop_last else math.ceil(len(self.train_index[self.data_key])/self.batch_size)
            # total = int(self.data_index["train_index_end"][self.data_key] / self.batch_size) if drop_last else math.ceil(self.data_index["train_index_end"][self.data_key] / self.batch_size)
            # print(total, self.data_index["train_index_end"][self.data_key] / self.batch_size)
            np.random.shuffle(self.train_index[self.data_key])
            for batch in range(total):
                yield self.get_data(self.train_index[self.data_key][batch*self.batch_size:(batch+1)*self.batch_size])

                #return 
        if (batch_size == -1):
            return callback
        else:
            return self.get_8x8_from_2048( batch_size, self.train_index , drop_last)

    def test_iter(self, drop_last = True,  batch_size = -1):
        def callback():
            total = int( len(self.test_index[self.data_key])/self.batch_size ) if drop_last else math.ceil(len(self.test_index[self.data_key])/self.batch_size)
            # iter_total = int( total / self.batch_size) if drop_last else math.ceil(total / self.batch_size)
            #p rint(total, iter_total)
            np.random.shuffle(self.test_index[self.data_key])
            for batch in range(total):
                yield self.get_data( self.test_index[self.data_key][batch*self.batch_size:(batch+1)*self.batch_size])
                #return
        if (batch_size == -1):
            return callback
        else:
            return self.get_8x8_from_2048( batch_size, self.test_index, drop_last )

    def val_iter(self, drop_last = True, batch_size = -1):
        def callback():
            total = int( len(self.val_index[self.data_key])/self.batch_size ) if drop_last else math.ceil(len(self.val_index[self.data_key])/self.batch_size)
            # iter_total = int( total / self.batch_size) if drop_last else math.ceil(total / self.batch_size)
            np.random.shuffle(self.val_index[self.data_key])
            for batch in range(total):
                yield self.get_data(self.val_index[self.data_key][batch*self.batch_size:(batch+1)*self.batch_size]) 
        
        if (batch_size == -1):
            return callback
        else:
            return self.get_8x8_from_2048( batch_size, self.val_index, drop_last )

    def __len__(self):
        return self.source_file_len[self.data_key+"_len"]
    
    def __iter__(self):
        return self
       
    def __del__(self):
        print("释放内存")
        for hdffile in self.source_file:
            hdffile.close()
            #print(hdffile)

    def printname(self, name):
        print(name)

    def split_train_and_test(self, shuffle = True, train_size = 0.9, keep_same = True):
        #self.data_index["2048"] = np.arange(self.source_file_len["2048_len"])
        #self.data_index["256"] = np.arange(self.source_file_len["256_len"])

        for key in self.train_index.keys(): # ["256","2048"]:
            yes = []
            no = []
            
            for index in range( len(self.source_file_len[key])-1 ):
                file_index = np.arange(self.source_file_len[key][index], self.source_file_len[key][index+1], dtype = np.int)
                #if(shuffle):
                    #np.random.shuffle( file_index )
                for middle_index in file_index:
                    origin_index = middle_index - self.source_file_len[key][index]
                    if(self.source_file[index][key]["label"][origin_index]):
                        yes.append(middle_index)
                    else:
                        no.append(middle_index)
                
                #total_number  = len(file_index)
                
                
                #train_end = int( train_size * total_number)
                #temp_index = int( (total_number - train_end)/2 )
                #val_end = train_end + temp_index
                
                #self.train_index[key] = np.concatenate((self.train_index[key], file_index[:train_end])) 
                #self.val_index[key] = np.concatenate((self.val_index[key], file_index[train_end:val_end]))
                #self.test_index[key] = np.concatenate(( self.test_index[key],file_index[val_end:]))
                
                
            total_yes = len(yes)
            total_no = len(no)

            total_yes_train_end = int( train_size * total_yes)
            total_no_train_end = int( train_size * total_no)
            self.total_yes_train_end[key] = total_yes_train_end
            self.total_no_train_end[key] = total_no_train_end
            #temp_yes_index = int( (total_yes - total_yes_train_end))
            #temp_no_index = int( (total_no - total_no_train_end))    
            #val_yes_end = total_yes_train_end + temp_yes_index
            #val_no_end = total_no_train_end + temp_no_index                
            if(keep_same):
                limit = np.min((len(yes),len(no)))
                print(len(yes), len(no),":->")
                no = no[:limit]
                yes = yes[:limit]
                print("->:",len(yes), len(no))
            np.random.shuffle( yes )
            np.random.shuffle( no )
            self.train_index[key] = np.concatenate((self.train_index[key], yes[:total_yes_train_end], no[:total_no_train_end]))
            self.test_index[key] = np.concatenate((self.test_index[key], yes[total_yes_train_end:], no[total_no_train_end:]))
            self.val_index[key] = self.test_index[key]
            if(shuffle):
                np.random.shuffle( self.train_index[key] )
                np.random.shuffle( self.val_index[key] )
                np.random.shuffle( self.test_index[key] )
        
        logging.info("2048 train size:{}, val size:{}, test size:{}\n256 train_size:{}, val size:{}, test size:{}\n".format(
            len(self.train_index["2048"]),

            len(self.val_index["2048"]),

            len(self.test_index["2048"]),

            len(self.train_index["256"]),

            len(self.val_index["256"]),

            len(self.test_index["256"]),

        ))
        print("2048 train size:{}, val size:{}, test size{}, \n256 train_size:{}, val size:{}, test size:{}\n".format(
            len(self.train_index["2048"]),

            len(self.val_index["2048"]),

            len(self.test_index["2048"]),

            len(self.train_index["256"]),

            len(self.val_index["256"]),

            len(self.test_index["256"]),
        ))

    #@log(TYPE_TEST, "load h5")                
    def load_h5(self, files_path):
        print("load path:",files_path,"\n")
        logging.info("load path:"+files_path)
        hf = h5py.File(files_path, "r")
        
        
        #print("dict value:")
        #hf.visit(self.printname)
        self.source_file.append(hf)
        total_2048 = len(hf["2048"]["image"])
        total_256 = len(hf["256"]["image"])

        self.source_file_len["256"].append(self.source_file_len["256"][-1] + total_256)
        self.source_file_len["2048"].append(self.source_file_len["2048"][-1] + total_2048)

        self.source_file_len["2048_len"] += total_2048
        self.source_file_len["256_len"] += total_256
        self.offset.append( hf["offset"] )
        # hf["2048"]["image"]
        # hf["2048"]["label"]
        # hf["2048"]["coord"]
        # hf["2048"]["index"]

        # hf["256"]["image"]
        # hf["256"]["label"]
        # hf["256"]["coord"]
        # hf["256"]["index"]

    @property
    def key(self):
        print("data dict key list:")
        for key, value in enumerate( self.data_set_scale ):
            print("key:",value)
    
    @key.setter
    def setkey(self, key_value):
        self.data_key = key_value
    
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
            
            if(down + top + right + left + right_top + right_down + left_top + left_down == 0): #孤立�?
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

    #废弃
    def coding_absence_old(self, data , show = True, vector = None, label = 0, data_type = "coord"):
        image = np.zeros((10,10))
        #     image[:,0] = 1
        #     image[0,:] = 1
        #     image[-1,:] =1
        #     image[:,-1] = 1
        crop_image_list = data["256"]
        total = len(crop_image_list)

        for index in range(total):
            i = crop_image_list[index]
            if(data_type == "coord"):
                row_256_s, row_256_e, col_256_s, col_256_e = i["coord"]
                row_256_s = row_256_s%2048  
                #col_256_s = col_256_s/256
                if(vector is not None):
                    image[ int(row_256_s/256) +1, int(col_256_s/256) +1] = vector[index]
                else:
                    image[ int(row_256_s/256) +1, int(col_256_s/256) +1] = i["label"]
            else:
                image[1:9,1:9] = vector.reshape(8,8)
        if(show):
            plt.imshow( image )
            plt.show()
        return image
    #废弃
    def coding_absence(self, data , show = True, vector = None, label = 0):
        image = np.zeros((10,10))
        image[1:9,1:9] = vector.reshape(8,8)
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
                    temp["256"].append(j["image"])
            data.append(temp)
        return data
    
    def get_group_by_index_old(self, file_index, group_index = None):
        file_ = self.source_file[file_index]
        if group_index is None:
            group_index = self.source_file[file_index]["2048"]["index"][0]
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
                    "map": file_["2048"]["map"][index],
                }
                data_dict["2048"].append(i)
        return data_dict
    
    def get_group_by_index(self, file_index, group_index = None, easy_group = True, normal = True):
        file_ = self.source_file[file_index]
        total = len(file_["2048"]["image"])
        data_dict = {"256":[],"2048":[]}
        if(easy_group):
            for index in range(total):
                source_index = file_["2048"]["index"][index]
                image = file_["2048"]["image"][index]
                if normal :
                    image = copy.deepcopy(image - np.mean(image, 0))
                    #image = (image - np.mean(image))/np.std(image)
                #print(image.dtype, image.shape)
                i = {
                        "image":image,
                        "coord":file_["2048"]["coord"][index],
                        "label":file_["2048"]["label"][index],
                        'map':file_["2048"]["map"][index],
                        "index":source_index,
                    }
                data_dict["2048"].append(i)
            return data_dict
            
        if group_index is None:
            group_index = self.source_file[file_index]["2048"]["index"][0]
            #print(file_)
        
        
        for index in range(total):
            source_index = file_["2048"]["index"][index]
            if(source_index == group_index):
                i = {
                        "image":file_["2048"]["image"][index],
                        "coord":file_["2048"]["coord"][index],
                        "label":file_["2048"]["label"][index],
                        'map':file_["2048"]["map"][index],
                        "index":source_index,
                    }
                data_dict["2048"].append(i)
        return data_dict
    
    def save_data(self, save_name, data, save = True):
        dpi = 128
        height, width = data.shape
        if(height <= 50 and width <= 50):
            height = 100
            width = 100
        if(height >= 1024 and width >= 1024):
            dpi = 256
            height = 1024
            width = 1024
        plt.figure(figsize=(1.2993*height/dpi, 1.2993*width/dpi), dpi=dpi)
        plt.axis('off')
        plt.imshow(data)
        if(save):
            plt.savefig(save_name, bbox_inches='tight',pad_inches=0.0,dpi = dpi)
        plt.clf()
        plt.close('all')
        
    def save_as_h5(self, file_path, name):  
        import h5py
        f = h5py.File(file_path + name + "0.h5", "w")
        print(file_path+ name + "0.h5")
        image_list = {
            "yes":[],
            "no":[],
        }
        
        for i in range(len(self.source_file)):
            print("start process index:",i)
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
                        middle_name = "yes/"    
                    else:
                        image_list["no"].append(image)
                        middle_name = "no/"    
                    name = file_path + "image/"+ middle_name + str(fits_index)+ "_" +str(index) +".png"
                    self.save_data(name, image)
                    name = file_path + "image/"+ middle_name + str(fits_index)+ "_" +str(index) +"_origin.png"
                    self.save_data(name,  data_group["2048"]["image"])
                    
                index += 1

        print(len(image_list["yes"]), len(image_list["no"]))
        #g = f.create_group("data")
        f.create_dataset("data_yes", data=copy.deepcopy(image_list["yes"]))
        f.create_dataset("data_no", data=copy.deepcopy(image_list["no"]))
        f.close()
        
    def crop_map_for_predict(self, file_path, data_type = 0, need_rot =False, normal = False, need_origin = False):
        #filename = self.copyfile( file_path, out_dir +"/")
        hdulist = pyfits.open(file_path)
        data_list_2048 = []
        try:
            hdu0 = hdulist[0] 
            hdu1 = hdulist[1]
            tsamp = hdu1.header['TBIN']
            data1 = hdu1.data['data']
            obsnchan = hdu0.header['OBSNCHAN']

            if len(data1.shape)>2:
                a,b,c,d,e = data1.shape
                data = data1[:,:,0,:,:].squeeze().reshape((-1,d))
                l, m = data.shape
                total_time = tsamp*l
            else:
                a,b = data1.shape 
                data = data1.reshape(-1,int(obsnchan))
                l,m=data.shape
            
            key = m
            sub_num = int(l/key)
            subt_num = int(m/key)
            print("read file:",file_path,",file shape:({},{})".format(l,m))
            if need_origin:
                origin_data = copy.deepcopy(data)
                if(key*sub_num != l):
                    origin_data_list = np.split(origin_data[:key*sub_num], sub_num, axis = 0)
                    origin_data_list.append(data[-key:])
                else:
                    origin_data_list = np.split(origin_data, sub_num, axis = 0)
            if(need_rot):
                data = np.rot90(data,2)
            if(normal):
                data = (data - np.mean(data,0))
                data = np.where(data>=1, 255, 0)
                # data[ data < 1 ] = 0
                # data[ data > 1 ] = 255
            
            data_list_2048_label = []
            if(key*sub_num != l):
                data_list_2048 = np.split(data[:key*sub_num], sub_num, axis = 0)
                data_list_2048.append(data[-key:])
            else:
                data_list_2048 = np.split(data, sub_num, axis = 0)
                
            #data_list_2048 = [ data_2048  for data_2048 in data_list_2048_col]
            for j in range(subt_num):
                for i in range(sub_num):
                    save_name = "{}_{}.jpg".format(i,j)
                    data_list_2048_label.append(save_name)
            if(key*sub_num != l):
                data_list_2048_label.append("{}_{}_add.jpg".format(i,j))
            data_list_2048 = [ data_list_2048, data_list_2048_label]

            # data_list_2048 = []
            # data_list_2048_label = []

            # for j in range(subt_num):
            #     for i in range(sub_num):

            #         image = data[i*key:(i+1)*key,j*key:(j+1)*key]
            #         save_name = "{}_{}.jpg".format(i,j)
            #         data_list_2048.append(image)
            #         data_list_2048_label.append(save_name)
            # data_list_2048 = [ data_list_2048, data_list_2048_label]
            
                
            # for rawi in range(sub_num):
            #     for colj in range(subt_num): 
            #         r_start_index , r_end_index = rawi * 2048, (rawi+1) * 2048
            #         c_start_index , c_end_index = colj * 2048, (colj+1) * 2048
            #         data_list_2048.append(copy.deepcopy(data[r_start_index:r_end_index, c_start_index:c_end_index]))
             
        except Exception as e:
            print(e)
            return None
        hdulist.close()
        if need_origin:
            return data_list_2048, origin_data_list
        else:
            return data_list_2048

        
    def crop_map_for_predict_OTSU(self, file_path, data_type = 0, need_rot =True, normal = False):
        #filename = self.copyfile( file_path, out_dir +"/")
        hdulist = pyfits.open(file_path)
        data_list_2048 = []
        try:
            hdu0 = hdulist[0] 
            hdu1 = hdulist[1]
            obsbw = hdu0.header['OBSBW']
            tsamp = hdu1.header['TBIN']
            data1 = hdu1.data['data']
            obsfreq = hdu0.header['OBSFREQ']
            fmin = obsfreq - obsbw/2.
            fmax = obsfreq + obsbw/2.
            fchannel = hdulist['SUBINT'].data[0]['DAT_FREQ']
            fchn = len(fchannel)
            obsnchan = hdu0.header['OBSNCHAN']

            if len(data1.shape)>2:
                a,b,c,d,e = data1.shape
                data = data1[:,:,0,:,:].squeeze().reshape((-1,d))
                l, m = data.shape
                total_time = tsamp*l
            else:
                a,b = data1.shape 
                data = data1.reshape(-1,int(obsnchan))
                l,m=data.shape

            key = m
            sub_num = int(l/key)
            subt_num = int(m/key)
            print("read file:",file_path,",file shape:(%d,%d)".format(l,m))
            if(need_rot):
                data = np.rot90(data,2)
            if(normal):
                data = data - np.array(np.mean(data,0),dtype = np.uint8)
                ret, data  = cv2.threshold(data, 0,1, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
              
            data_list_2048_label = []
            data_list_2048 = np.split(data, sub_num, axis = 0)
            #data_list_2048 = [ data_2048  for data_2048 in data_list_2048_col]
            for j in range(subt_num):
                for i in range(sub_num):
                    save_name = "{}_{}.jpg".format(i,j)
                    data_list_2048_label.append(save_name)
            data_list_2048 = [ data_list_2048, data_list_2048_label]
            
                
            # for rawi in range(sub_num):
            #     for colj in range(subt_num): 
            #         r_start_index , r_end_index = rawi * 2048, (rawi+1) * 2048
            #         c_start_index , c_end_index = colj * 2048, (colj+1) * 2048
            #         data_list_2048.append(copy.deepcopy(data[r_start_index:r_end_index, c_start_index:c_end_index]))
             
        except Exception as e:
            print(e)
            return None
        hdulist.close()
        return data_list_2048



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

    

