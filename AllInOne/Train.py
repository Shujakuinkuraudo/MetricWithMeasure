import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from AllInOne.Loss import ContrastiveLoss
from AllInOne.dataset import BirdClefDataset, ButterflyDataset, CoronavirusDataset, CIFAR100
from AllInOne.Model import Model
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import datetime


class Train:
    def __init__(self, time) -> None:
        self.time = time
        self.create_folder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_contrastive = ContrastiveLoss().to(self.device)
        self.loss_ae = nn.MSELoss().to(self.device)
        self.dataset = {
            True: {
                "Cifar100": ("Image", CIFAR100()),
                "Bird": ("Audio", BirdClefDataset()),
                "Coron": ("Text", CoronavirusDataset()),
                "Butterfly": ("Image", ButterflyDataset()),
            },
            False: {
                "Cifar100": ("Image", CIFAR100(False)),
                "Butterfly": ("Image", ButterflyDataset(False)),
                "Coron": ("Text", CoronavirusDataset(False)),
                "Bird": ("Audio", BirdClefDataset(False)),
            },
        }
        self.model = Model().to(self.device)
        self.optim = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": 1e-3},
                {"params": self.loss_contrastive.parameters(), "lr": 1e-4},
            ]
        )
        self.epochs = 1000

    def create_folder(self):
        import os
        if not os.path.exists(f"fig/{self.time}"):
            os.mkdir(f"result/{self.time}")
        os.mkdir(f"result/{self.time}/fig")
        os.mkdir(f"result/{self.time}/model")

    def save(self, epoch):
        torch.save(self.model.state_dict(), f"result/{self.time}/model/{epoch}.pth")

    def load(self, pth):
        self.model.load_state_dict(torch.load(pth))

    def get_loader(self, dataset_name, train=True):
        type, dataset = self.dataset[train][dataset_name]
        return type, DataLoader(
            dataset,
            batch_size=128,
            shuffle=train,
            num_workers=16,
            pin_memory=True,
            prefetch_factor=3,
        )

    def train_one_Loader(self, loader, type):
        self.model.train()
        losses_ae = []
        losses_contrastive = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            x, pred = self.model(x, type)
            loss = (loss_ae := self.loss_ae(pred, x)) + (
                loss_contrastive := self.loss_contrastive(pred.view(x.size(0), -1), y)
            )
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            losses_ae.append(loss_ae.detach().item())
            losses_contrastive.append(loss_contrastive.detach().item())
        return sum(losses_contrastive) / len(losses_contrastive), sum(losses_ae) / len(
            losses_ae
        )

    def train_all(self):
        for epoch in tqdm(range(self.epochs)):
            for dataset_name, (type, dataset) in list(self.dataset[1].items()):
                type, loader = self.get_loader(dataset_name)
                loss_con, loss_ae = self.train_one_Loader(loader, type)
                print(loss_con, loss_ae)
                self.log(f"{epoch}\t dataset:{dataset_name}\t loss_con:{loss_con} \tloss_ae:{loss_ae} \t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                if epoch % 10 == 0:
                    self.visualize(
                        dataset_name, output=f"result/{self.time}/fig/{dataset_name}_{epoch}.png"
                    )
            if epoch % 10 == 0:
                self.save(epoch)

    def log(self,string):
        with open(f"result/{self.time}/log.txt","a") as f:
            f.write(string+"\n")
    
    def train_one(self, dataset_name):
        for epoch in tqdm(range(self.epochs)):
            type, loader = self.get_loader(dataset_name)
            loss_con, loss_ae = self.train_one_Loader(loader, type)
            self.log(f"{epoch}\t dataset:{dataset_name}\t loss_con:{loss_con} \tloss_ae:{loss_ae} \t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if epoch % 10 == 0:
                self.visualize(
                    dataset_name, output=f"result/{self.time}/fig/{dataset_name}_{epoch}.png"
                )
            if epoch % 10 == 0:
                self.save(epoch)

    def eval(self, samples, ys):
        kmeans = KMeans(len(np.unique(ys)), n_init="auto")
        labels = kmeans.fit_predict(samples)
        score = adjusted_mutual_info_score(ys, labels)
        self.log(f"cluster_score:{score}")
        print("cluster_score", score)

    def test(self, dataset_name):
        self.model.eval()
        type, loader = self.get_loader(dataset_name, train=False)
        samples = []
        ys = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, pred = self.model(x, type)
            pred = pred.reshape(pred.size(0), -1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            for i in range(pred.shape[0]):
                samples.append(pred[i])
                ys.append(y[i])
        return samples, ys

    def visualize(self, dataset_name, output=None):
        from sklearn.decomposition import PCA
        from matplotlib.pyplot import scatter
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.decomposition import PCA

        samples, ys = self.test(dataset_name)
        self.eval(samples, ys)

        samples_PCA = PCA(n_components=3).fit_transform(samples)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(samples_PCA[:, 0], samples_PCA[:, 1], samples_PCA[:, 2], c=ys)
        plt.show()
        if output:
            plt.savefig(output)
