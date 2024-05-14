# TFR

TFR is frb classfiy deeplearning method. is our paper "Efficient FRB Signal Search with Two-Stage Weak Signal Feature Reconstruction" publish code

# Environment

```
GPUtil
numpy==1.21.5
torch==1.12.0
opencv-python
astropy #using conda install astropy to install
h5py
```

# Dataset

We have produced simulation datasets of different specifications and labeled real datasets, please send email to 407157175@qq.com

# Train 

run python train.py in cmd with config

```
batch_size = 32
train_dir_path = "Path/To/DataSet"
data_shape = (4096, 4096)
method_dict ={
    "conv17":0,
    "inceptionresnetv2":1,
    "dense121":2,
    "efficientnet":3,
    "cmt":4,
    "ADFP":5,
    "PCT":6,
}

trainer = Train(
    image_shape = data_shape,
    class_number = 2, 
    is_show = False,
    name = "PCT",
    method_type = 6
)
```

run python ./TFR/model/torch_linear/train.py in cmd for GA model and DDPM Score Match



# Predict

run run_predict.py  with parameters

```python
'--model', default=2, choices=[0, 1, 2, 3, 4]
 ## Name of model to predict
 ## 0 -- conv17
 ## 1 -- inceptionresnetv2
 ## 2 -- dense121
 ## 3 -- efficientnet
 ## 4 -- cmt')
 
'--save_path', default='./predict_ans/'
 ## 'predict data save path'

'--data_dir', default="/home/data/lijl/DATA/Frbdata/fast/", 
## 'data dir path'

'--need_code', default=False, type=bool, 
## 'save map code and save print data'
```

# Log

Please refer to log folder for our experimental parameters and log results

# Effectiveness

| <img src=".\log\hotmap\4_0.jpg" alt="4_0" style="zoom:33%;" /> | <img src=".\log\hotmap\4_0_mask.jpg" alt="4_0_mask" style="zoom:150%;" /> |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| <img src=".\log\hotmap\11.jpg" alt="11" style="zoom:150%;" /> | <img src=".\log\hotmap\11_mask.jpg" alt="11_mask" style="zoom:150%;" /> |      |
| <img src=".\log\hotmap\7_0.jpg" alt="7_0" style="zoom:33%;" /> | <img src=".\log\hotmap\7_0_mask.jpg" alt="7_0_mask" style="zoom:150%;" /> |      |
| <img src=".\log\hotmap\28_0.jpg" alt="28_0" style="zoom:33%;" /> | <img src=".\log\hotmap\28_0_mask.jpg" alt="28_0" style="zoom:150%;" /> |      |

