import argparse
from pathlib import Path
import os
from predict_v2_2 import *
import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('TGAC predict script', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default=2, choices=[0, 1, 2, 3, 4] ,type=int, metavar="0,1,2,3,4",
                        help='## Name of model to predict\n## 0 -- conv17\n## 1 -- inceptionresnetv2\n## 2 -- dense121\n## 3 -- efficientnet\n## 4 -- cmt\n')

    parser.add_argument('--judge_method_name', default='GA_pipline', type=str, metavar='GA_pipline',
                        help='judge_method_name')

    parser.add_argument('--save_path', default='./predict_ans/', type=str, metavar='/home/save_path/',
                        help='predict data save path')

    parser.add_argument('--GA_model_path', default="./model/torch_linear/param/origin/best.pkl", type=str, metavar='/home/save_path/',
                        help='GA_model save path')
    
    parser.add_argument('--data_dir', default="/home/data/lijl/DATA/Frbdata/Wang/origin/", type=str, metavar='/home/data path/',
                        help='data dir path')

    parser.add_argument('--need_code', default=False, type=bool, help='save all map code')

    parser.add_argument('--save_candidate', default=True, type=bool, help='only save candidate map code in name floder')


    parser.add_argument('--start', default=0, type=int, metavar='start file index',
                        help='start search file  index')
    parser.add_argument('--end', default=-1, type=int, metavar='end file index',
                        help='end search file  index')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--quantization', default = False, type = bool,
                        help='using quantization model to predict only cpu, almost enchence inference speed 50%')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--name', default="dense", type=str)
    
    return parser

if __name__ == '__main__':

    parameter_path_dict = { 
        0:"./save_best/conv17_8x8_2048/best.pkl",
        1:"./save_best/inceptionresnetv2_8x8_2048/best.pkl",
        2:"./save_best/dense121_8x8_2048/best.pkl",
        3:"./save_best/efficientnet_8x8_2048/best.pkl",
        4:"./save_best/cmt_8x8_2048_origin/best.pkl",
    }

    parser = argparse.ArgumentParser('TGAC predict script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    data_dir = args.data_dir
    method_type = args.model
    save_path = args.save_path
    GA_model_path = args.GA_model_path
    judge_method_name = args.judge_method_name
    need_code = args.need_code
    device = args.device
    seed = args.seed
    name = args.name
    save_candidate = args.save_candidate
    start = args.start
    end = None if args.end == -1 else args.end 
    data_split = data_dir.split('/')
    date = datetime.datetime.now()
   
    quantization = args.quantization 
    log_name = ""
    for i in data_split:
        log_name += i.replace("-",'_') + "_"
    log_path = save_path +  str(date)[:10].replace('-',"") + log_name + name + ".txt"
    image_save_path = save_path + "/" + name + "/" 
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    image_save_path = image_save_path + data_split[-2] + "/"
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('image_save_path: %s'%(image_save_path))
    print('log_path: %s'%(log_path))
    print('model path: %s'%(parameter_path_dict[method_type]))
    print('--------args----------\n')
    
    detect_pipline = FRB_CA_Pipline(data_dir, name, method_type=method_type, parameter_path_dict=parameter_path_dict, CA_model_path = GA_model_path, save_candidate = save_candidate, quantization = quantization)
    file_path_list = detect_pipline.dg.train_file_path_list
    detect_pipline.detect_fits_list(log_path ,file_path_list,start = start, end = end, data_type =  1, need_save = need_code, need_code = need_code, image_save_path = image_save_path)
    print('-----end search-------')
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('image_save_path: %s'%(image_save_path))
    print('log_path: %s'%(log_path))
    print('model path: %s'%(parameter_path_dict[method_type]))
    print('--------args----------\n')



