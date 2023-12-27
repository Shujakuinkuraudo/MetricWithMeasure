import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from AllInOne.Train import Train
import datetime

t = Train(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
t.measure_train_all("result/2023-12-26-19-59-20 */model/190.pth")