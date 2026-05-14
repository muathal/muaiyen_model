from preproccess.Preproccess import *
from Model.Model import SGM
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
torch.manual_seed(42)
train = pd.read_json("Data/train.json")
val = pd.read_json("Data/val.json")
test = pd.read_json("Data/test.json")
text_mapping,text_unmapping = build_vocab(train["text"])
slot_mapping,intent_mapping,slot_unmapping,intent_unmapping = build_mapping(train)
train_data = ERPDataset(train,text_mapping,slot_mapping,intent_mapping)


