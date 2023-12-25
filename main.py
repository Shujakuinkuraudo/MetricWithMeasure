import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import matplotlib.pyplot as plt
import AllInOne.dataset
from AllInOne.Train import Train
from AllInOne.dataset import Dataset
import datetime
import torch

t = Train(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
t.train_all()