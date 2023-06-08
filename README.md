# TFR

TFR is frb classfiy deeplearning method. Global Attention Connect with two stage model for FRB classfiy

# Environment

```
GPUtil
numpy==1.21.5
torch==1.12.0
opencv-python
astropy #using conda install astropy to install
h5py
```

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

