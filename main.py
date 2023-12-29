import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from AllInOne.Train import Train
import datetime

t = Train(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
# t.measure_train_all("result/2023-12-26-19-59-20 */model/190.pth")
t.metric_train_all("result/2023-12-27-07-48-02/model/190.pth")