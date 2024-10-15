import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict

from sklearn.model_selection import train_test_split
from sklearn import metrics 
import math

import os
import cv2
import torch
import joblib
import timm
import torch.nn as nn
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.regression import R2Score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models import efficientnet
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import sys
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict

def print_uniques(df:pd.DataFrame):
    for col in df.columns:
        print(col)
        print(df[col].unique())
        
def encode_columns(df:pd.DataFrame, encoding_cols:list[str]) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    for col in encoding_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
    return df

def drop_cols(df:pd.DataFrame, dropped_cols:list[str]) -> pd.DataFrame:
    df.drop(columns=dropped_cols, inplace=True)
    return df

def transform_year(df:pd.DataFrame, sub_by:int) -> pd.DataFrame:
    df["model_year"] = df["model_year"].apply(lambda x : x - 1992)
    return df

def transform_milage_price(df:pd.DataFrame) -> pd.DataFrame:
    if "price" in df.columns:
        df["price"] = df["price"].apply(lambda x: math.log(x))
    if "milage" in df.columns:
        df["milage"] = df["milage"].apply(lambda x: math.log(x))
    return df

def load_data(path:str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def augment_df(df:pd.DataFrame) -> pd.DataFrame:
    dropped_cols = ["id"]
    df = drop_cols(df, dropped_cols=dropped_cols)
    df = encode_transmissions(df)
    columns_to_encode = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident", "clean_title"]
    df = encode_columns(df, columns_to_encode)
    df = transform_milage_price(df)
    return df

def encode_transmissions(df:pd.DataFrame) -> pd.DataFrame:
    df["transmission_type"] = df["transmission"].apply(lambda x:
                                                  'manual' if 'm/t' in x or 'manual' in x or  'mt' in x else 
                                                  'automatic' if 'a/t' in x or 'automatic' in x or  'at' in x else 
                                                  'CVT' if 'CVT' in x else 
                                                  'Other')
    
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    df["transmission_type"] = enc.fit_transform(df["transmission"])
    
    return df

def train_NN(df:pd.DataFrame):
    from RegDataloader import RegDataloader
    dataset = RegDataloader(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    from RegModels import SimRegModel
    model = SimRegModel(in_size = 11, out_size=1)
    device="cuda"
    model.to(device)
    Epochs = 10
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    criterion = nn.MSELoss()
    
    for epochs in range(Epochs):
        model.train()
        total_train_loss = 0.0
        train_batches = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            input_data = x.to(device, dtype=torch.float32)
            target = y.to(device, dtype=torch.float32)
            
            out = model(input_data)
            size = out.shape[0]
            loss = torch.sqrt(criterion(out.reshape(size), target))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_batches += 1
            
        print(f"Epoch: {epochs+1}/{Epochs}, Average Train RMSE Loss: {total_train_loss / train_batches}")

def train_RF(df:pd.DataFrame, 
             test_df:pd.DataFrame = None, 
             submission_df:pd.DataFrame = None, 
             test:bool = False) -> Union[float, np.array]:
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    rf_model = RandomForestRegressor(n_estimators=1000, verbose=3, n_jobs=10, random_state=987)
    

    if not test:
        xtrain, xtest, ytrain, ytest = train_test_split(df[df.columns[:-1]].values, df[df.columns[-1]], test_size=0.3)
        
        rf_model.fit(xtrain, ytrain)
        ypred = rf_model.predict(xtest)
        mse = mean_squared_error(ytest, ypred)
        rmse = math.sqrt(mse)
        print(f"MSE: {mse}, RMSE: {rmse}")
        return rmse
    else:
        rf_model.fit(df[df.columns[:-1]].values, df[df.columns[-1]])
        pred = rf_model.predict(test_df.values)
        submission_df["price"] = pred
        print(pred)
        submission_df["price"] = submission_df["price"].apply(lambda x: math.exp(x))
        print(submission_df)
        submission_df.to_csv("Submissions/RandomForest_Submission.csv", index=False)
        
def train_SVM(df:pd.DataFrame, 
             test_df:pd.DataFrame = None, 
             submission_df:pd.DataFrame = None, 
             test:bool = False) -> Union[float, np.array]:
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error

    #poly might work well as the values correlate with one another but more functionally and less radially
    svm_model = SVR(C=1, kernel="poly", degree=9, verbose=True)
    
    if not test:
        xtrain, xtest, ytrain, ytest = train_test_split(df[df.columns[:-1]].values, df[df.columns[-1]], test_size=0.3)
        
        svm_model.fit(xtrain, ytrain)
        ypred = svm_model.predict(xtest)
        mse = mean_squared_error(ytest, ypred)
        rmse = math.sqrt(mse)
        print(f"MSE: {mse}, RMSE: {rmse}")
        return rmse
    else:
        svm_model.fit(df[df.columns[:-1]].values, df[df.columns[-1]])
        pred = svm_model.predict(test_df.values)
        submission_df["price"] = pred
        print(pred)
        submission_df["price"] = submission_df["price"].apply(lambda x: math.exp(x))
        print(submission_df)
        submission_df.to_csv("Submissions/SVM_Submission.csv", index=False)

def main():
    train_path = "~/Documents/Kaggle-comps/Used_Car_Regression/Data/playground-series-s4e9/train.csv"
    test_path = "~/Documents/Kaggle-comps/Used_Car_Regression/Data/playground-series-s4e9/test.csv"
    submission_path = "~/Documents/Kaggle-comps/Used_Car_Regression/Data/playground-series-s4e9/sample_submission.csv"
    df = load_data(path=train_path)
    df = augment_df(df)
    test_df = load_data(path=test_path)
    test_df = augment_df(test_df)
    submission_df = load_data(path=submission_path)
    submission_df.drop(columns="price", inplace=True)
    # print(submission_df)
    # train_NN(df)
    train_RF(df, test_df, submission_df, True)
    # train_SVM(df)
    # print(df[df.columns[:-1]])
    # print(df[df.columns[-1]])
    
def test():
    train_path = "~/Documents/Kaggle-comps/Used_Car_Regression/Data/playground-series-s4e9/train.csv"
    test_path = "~/Documents/Kaggle-comps/Used_Car_Regression/Data/playground-series-s4e9/test.csv"
    submission_path = "~/Documents/Kaggle-comps/Used_Car_Regression/Data/playground-series-s4e9/sample_submission.csv"
    df = load_data(path=train_path)
    # df = augment_df(df)
    print_uniques(df)
    
    
def test2():
    k = [1,2,3,4,5]
    # k2 =[2,4,6,8,10]
    k3 =[100,141,413431231,32,321,321321]
    
    
    
if __name__ == '__main__':
    main()
    # test()