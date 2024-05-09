import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import model_selection
import random
import os
from engine import Engine
from utils import CustomDataset
from model import Model
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

Device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
def run_training(fold, save_model = False):
    df = pd.read_csv("../input/train_folds.csv")
    train_data = df[df.kfold != fold].reset_index(drop = True)
    val_data = df[df.kfold == fold].reset_index(drop = True)

    x_train = train_data.drop(['phishing','kfold'], axis = 1).values
    y_train= train_data['phishing'].values.astype(float)

    x_valid = val_data.drop(['phishing','kfold'], axis = 1).values
    y_valid = val_data['phishing'].values.astype(float)

    train_dataset = CustomDataset(x_train, y_train)

    valid_dataset = CustomDataset(x_valid, y_valid)

    train_loader = DataLoader(
        train_dataset, batch_size = 512, num_workers = 4, shuffle = True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size = 512, num_workers = 4, shuffle = False
    )

    cnn_filters = [128, 128, 128]
    cnn_kernel_sizes = [3, 4, 5]
    input_dim = 41
    hidden_dim = 64
    
    model = Model(input_dim, hidden_dim, cnn_filters, cnn_kernel_sizes)
    model.to(Device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-3)
    eng = Engine(model,optimizer,device = Device)

    best_acc = 0.0
    early_stopping_iter = 10
    early_stopping_counter = 0
    print(f"fold:{fold}")
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        val_loss,val_acc = eng.evaluate(valid_loader)

        print(f"epoch: {epoch+1}, training_loss :{round(train_loss,4)}, validation_loss: {round(val_loss,4)}, val_acc: {round(val_acc,4)}")
        if  val_acc > best_acc:
            best_acc = val_acc
            early_stopping_counter = 0
            if save_model:
                torch.save(model.state_dict(),f"../models/model_{fold}.bin")
        else:
            early_stopping_counter +=1

        if early_stopping_counter > early_stopping_iter:
            break
    return best_acc

if __name__ == "__main__":
    best_acc = []
    for fold in range(5):
        val_acc = run_training(fold, save_model=True)
        print(f"fold: {fold}, val_acc: {val_acc}")
        best_acc.append(val_acc)


    print(f"overall fold val acc: {np.mean(best_acc)}")