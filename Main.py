from preproccess.Preproccess import *
from Model.Model import SGM
import pandas as pd
train_data = ERPDataset(pd.read_json("Data/train.json"))
val_data = ERPDataset(pd.read_json("Data/val.json"))
test_data = ERPDataset(pd.read_json("Data/test.json"))

