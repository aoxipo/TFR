from train import Train
from Data_generate import Dataload
import matplotlib.pyplot as plt

if __name__ == "__main__":
    batch_size = 1
    train_dir_path = "F:/outpage/Virsualiz-torch-liear/data/" 
    dg = DataGenerate(train_dir_path, batch_size = batch_size)
    dg.split_train_and_test()
    trainer = Train(1, 2)
    trainer.load_parameter("./save/best.pkl")
    val_dataloader = []
    for i in dg.val_iter()():
        val_dataloader.append(i)
    
    
    ans = trainer.predict(val_dataloader)

    
    

    
    

    