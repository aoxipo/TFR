
from FRB_train import Train
from data_generate_h5 import DataGenerate
from model.torch_linear.train import Train as CA_Class 
import datetime
import numpy as np
from lib.util import *
import os



class FRB_CA_Pipline():
    def __init__(self,train_dir_path, name = "inceptionresnetv2" , method_type = 1, parameter_path_dict = None, CA_model_path = "./model/torch_linear/save/best.pkl") -> None:
        
        if parameter_path_dict is None:
            self.parameter_path_dict = { 
                0:"./save/conv17/best.pkl",
                1:"./save/inceptionresnetv2/best.pkl",
                2:"./save/dense121/best.pkl",
                3:"./save/efficientnet/best.pkl",
            }
        else:
            self.parameter_path_dict = parameter_path_dict
        self.normal = True
        self.image_shape = (4096,4096)
        self.dg = self.build_datagerate(train_dir_path)
        self.trainer = self.build_first_classifier(name, method_type)
        self.CA_model = self.build_CA_classifier(CA_model_path)
    
    # To do release the builder and class model
    def __del__(self):
        #print("wait for build del part")
        pass

    def get_data_dict(self, data_list, data_type = 0):
        crop_list = []
        if(data_type != 0):
            data_list = np.array(data_list[0])
        data_dict = {"2048":[]}
        for data_2048 in data_list:
            if(self.normal):
                data_2048 = data_2048 - np.mean(data_2048, 0)
                
            data_dict["2048"].append( data_2048 )
        return data_dict

    def test_one_fits_file(self, data_list, data_type = 0, need_code = False):
    
        data_dict = self.get_data_dict(data_list ,data_type = data_type)
        vector_256_list = self.trainer.predict_signle(data_dict)
        
        code_image_list = []
        for data_group_index in range(len(data_dict["2048"])):
            total_256 = len(vector_256_list[data_group_index])
            width = int(np.sqrt(total_256))
            code_image = np.zeros((2 + width, 2 + width ))
            code_image[1:(width+1),1:(width+1)] = vector_256_list[data_group_index].reshape(width,width)
            code_image_list.append(code_image)
            
        vector_2048_pred = self.CA_model.predict(np.array(code_image_list))  
     
        # low fitter
        # for i in range(len(code_image_list)):
        #     if(np.sum(code_image) > 15 ):
        #         vector_2048_pred[i] = 0
        if need_code:
            return vector_2048_pred.cpu().numpy(), code_image_list,
        else:
            return vector_2048_pred.cpu().numpy(), 

    def save_code(self, data_list, code_image_list, save_path = "./data/"):
        self.dg.makedir(save_path)
        
        total = len(data_list[0])
        for index in range(total):
            code_image = code_image_list[index]
            origin_image = data_list[0][index]
            if(self.normal):
                origin_image = origin_image - np.mean(origin_image, 0)
            save_name = data_list[1][index]
            save_code_name = "code_" + save_name
            self.dg.save_data(save_path + save_name, origin_image, save = True)
            self.dg.save_data(save_path + save_code_name, code_image, save = True)


    def detect_fits_list(self, save_path, file_path_list = None ,end = None, data_type = 0, need_save = False, need_code = False, image_save_path = None, need_rot = False):
        if(image_save_path != None):
            self.dg.makedir(image_save_path)
        f = open(save_path, "w+")
        if file_path_list is None:
            file_path_list = self.dg.train_file_path_list
        print("get file number total:",len(file_path_list))
        cost_time = []
        for fits_file_path in file_path_list[:end]:
            logging.info("process : {}".format(fits_file_path))
            start_time = datetime.datetime.now()

            data_list = self.get_crop_map(fits_file_path, data_type = data_type, need_rot = need_rot, normal=self.normal) 

            if(need_code):
                vector_2048_pred, code_image_list = self.test_one_fits_file(data_list, data_type = data_type, need_code = need_code)
            else:
                vector_2048_pred = self.test_one_fits_file(data_list, data_type = data_type)
            
            if(need_save and data_type):
                #print( fits_file_path.split("/")[-1])
                self.save_code(data_list, code_image_list, save_path=image_save_path + fits_file_path.split("/")[-1][:-5]+"/" )    
            cost_time.append((datetime.datetime.now() - start_time).seconds/len(vector_2048_pred)*64)
            have = np.sum(vector_2048_pred)
            log_str = "file name:{}, cost:{:.4f} second, have:{},|| ".format(
                fits_file_path, 
                cost_time[-1], 
                have,
            )

            print(log_str)
            logging.info(log_str)
            f.write(log_str)
            for pred in vector_2048_pred:
                f.write(str(pred)+",")
            f.write("\n")
        print("\n total: avg cost time for 64 * 4096: \n",np.mean(cost_time))
        f.close()

    def build_CA_classifier(self, parameter_path):
        CA = CA_Class(1,2, is_show = False)
        CA.load_parameter(parameter_path)
        return CA

    def build_datagerate(self, train_dir_path, out_path = "./data/"):
        start_time = datetime.datetime.now()
        dg = DataGenerate(train_dir_path = train_dir_path, file_type = "*.fits", data_shape=self.image_shape)
        end_time = datetime.datetime.now()
        print("cost time:",(end_time - start_time).seconds,"seconds")
        return dg

    def build_first_classifier(self, name, method_type):
        model = Train(
            image_shape = (256, 256),
            class_number = 2, 
            is_show = False,
            name = name,
            method_type = method_type,
        )
        path = self.parameter_path_dict[method_type]
        print("load param : ",path)
        model.load_parameter(path)
        return model

    # data_list_2048 = [ [ image:[[np.array]] ], [save name:str] ]
    def get_crop_map(self, file_path,  data_type = 0, need_rot = False, normal = True):
        #data_2048_list =  self.dg.crop_map_for_predict_OTSU(file_path, data_type = data_type, need_rot = need_rot, normal = normal)
        data_2048_list =  self.dg.crop_map_for_predict(file_path, data_type = data_type, need_rot = need_rot, normal = normal)
        return data_2048_list
    
    #特定fits
    def read_fits_by_index(self, file_path, index_list):
        data_2048_list = self.get_crop_map(file_path)
        ans_list = []
        for i in index_list:
            ans_list.append(data_2048_list[i])
        return ans_list






if __name__ == "__main__":

    #name = "efficientnet_8x8_2048"
    #name = "cmt_8x8_2048_origin"
    #name = "inceptionresnetv2_8x8_2048_transfer_same"
    #name = "efficientnet_8x8_2048_transfer"
    #name = "dense121_8x8_2048_OTSU_all"
    #name = "inceptionresnetv2_8x8_2048_transfer_hard_all_4"
    name = "conv17_8x8_2048_transfer_hard_all"
    
    method_type = 0
    parameter_path_dict = { 
            0:"./save_best/conv17_8x8_2048_transfer_all/best.pkl",
            1:"./save_best/inceptionresnetv2_8x8_2048_transfer_same/best.pkl",
            2:"./save_best/dense121_8x8_2048_OTSU_all//best.pkl",
            3:"./save_best/efficientnet_8x8_2048/best.pkl",
            #4:"./save_best/cmt_8x8_2048/best.pkl", # pocmt
            4:"./save_best/cmt_8x8_2048_origin/best.pkl",
            
        }
    
    parameter_path_dict_1 = { 
            0:"./save_best/conv17_8x8_2048_transfer_hard_all/best.pkl",
            1:"./save_best/inceptionresnetv2_8x8_2048_transfer_hard_all_4/best.pkl",
            2:"./save_best/dense121_8x8_2048_transfer_hard_all/best.pkl",
            3:"./save_best/efficientnet_8x8_2048_transfer_hard_all/best.pkl",
            #4:"./save_best/cmt_8x8_2048/best.pkl", # pocmt
            4:"./save_best/cmt_8x8_2048_origin_transfer_hard_all/best.pkl",
            
        }
    judge_method_name = "GA_pipline"
    CA_model_path = "./model/torch_linear/param/origin/best.pkl"
    save_root = "/home/data/lijl/DATA/Frbdata/Wang/origin_detect"
    if(not os.path.exists(save_root)):
        os.mkdir(save_root)
    save_path = "{}/{}/log_{}.txt".format(save_root,name, judge_method_name)
    
    data_dir = f"/home/data/lijl/DATA/Frbdata/Wang/origin/"
    image_save_path = save_root + "/" + name + "/"

    detect_pipline = FRB_CA_Pipline(data_dir, name, method_type=method_type,parameter_path_dict=parameter_path_dict, CA_model_path = CA_model_path)
    #method 1
    file_path_list = detect_pipline.dg.train_file_path_list
    #for i in file_path_list:
        #print(i)
    #print(file_path_list)
    detect_pipline.detect_fits_list(save_path ,file_path_list, data_type =  1, need_save = False, need_code = False, image_save_path = image_save_path)
    print(parameter_path_dict[method_type])