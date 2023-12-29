import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from AllInOne.Loss import ContrastiveLoss, MeasureLoss
from AllInOne.dataset import (
    BirdClefDataset,
    ButterflyDataset,
    CoronavirusDataset,
    CIFAR100,
)
from AllInOne.Model import MetricModel, MeasureModel
from tqdm import tqdm
import numpy as np
import datetime
import wandb


class Train:
    def __init__(self, time) -> None:
        self.time = time
        self.run = self.get_wandb()
        self.create_folder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_contrastive = ContrastiveLoss().to(self.device)
        self.loss_ae = nn.MSELoss().to(self.device)
        self.loss_measure = MeasureLoss().to(self.device)

        self.dataset = {
            True: {
                "Butterfly": ("Image", ButterflyDataset()),
                "Coron": ("Text", CoronavirusDataset()),
                "Cifar100": ("Image", CIFAR100()),
                "Bird": ("Audio", BirdClefDataset()),
            },
            False: {
                "Coron": ("Text", CoronavirusDataset(False)),
                "Cifar100": ("Image1", CIFAR100(False)),
                "Bird": ("Audio", BirdClefDataset(False)),
                "Butterfly": ("Image2", ButterflyDataset(False)),
            },
        }
        self.metric_model = MetricModel().to(self.device)
        self.measure_model = MeasureModel().to(self.device)
        self.optimizer = None
        self.epochs = 200

    def get_wandb(self):
        run = wandb.init(project="deep-learning", name="128batch", save_code=True)
        run.log_code("AllInOne/")
        return run

    def create_folder(self):
        import os

        if not os.path.exists(f"fig/{self.time}"):
            os.mkdir(f"result/{self.time}")
        os.mkdir(f"result/{self.time}/fig")
        os.mkdir(f"result/{self.time}/model")

    def save(self, epoch):
        torch.save(
            self.metric_model.state_dict(), f"result/{self.time}/model/{epoch}.pth"
        )

    def load(self, pth):
        self.metric_model.load_state_dict(torch.load(pth))

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

    def metric_train_one_Loader(self, loader, type):
        self.metric_model.train()
        losses_ae = []
        losses_contrastive = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            x, pred = self.metric_model(x, type)
            loss = (loss_ae := 0.1 * self.loss_ae(pred, x)) + (
                loss_contrastive := self.loss_contrastive(pred.view(x.size(0), -1), y)
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses_ae.append(loss_ae.detach().item())
            losses_contrastive.append(loss_contrastive.detach().item())
        return sum(losses_contrastive) / len(losses_contrastive), sum(losses_ae) / len(
            losses_ae
        )

    def metric_train_all(self, pth):
        if pth:
            self.metric_model.load_state_dict(torch.load(pth))
        self.optimizer = torch.optim.SGD(
            [
                {
                    "params": self.metric_model.FeatureExtraction.parameters(),
                    "lr": 1e-3,
                },
                {"params": self.metric_model.ae.parameters(), "lr": 1e-3},
            ]
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.98
        )

        for epoch in tqdm(range(self.epochs)):
            for dataset_name, (type, dataset) in list(self.dataset[1].items()):
                type, loader = self.get_loader(dataset_name)
                loss_con, loss_ae = self.metric_train_one_Loader(loader, type)
                print(loss_con, loss_ae)
                self.log(
                    f"{epoch}\t dataset:{dataset_name}\t loss_con:{loss_con} \tloss_ae:{loss_ae} \t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.run.log(
                    {
                        "epoch": epoch,
                        f"{dataset_name}_loss_con": loss_con,
                        f"{dataset_name}_loss_ae": loss_ae,
                    }
                )

                if epoch % 10 == 0:
                    self.visualize(
                        dataset_name,
                        output=f"result/{self.time}/fig/{dataset_name}_{epoch}.png",
                    )
            if epoch % 10 == 0:
                self.save(epoch)
            self.scheduler.step()

    def log(self, string):
        with open(f"result/{self.time}/log.txt", "a") as f:
            f.write(string + "\n")

    def measure_train_one_Loader(self, loader, type):
        self.measure_model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, feature = self.metric_model(x, type)
            pred = self.measure_model(feature)
            loss = self.loss_measure(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.detach().cpu().item())
        return sum(losses) / len(losses)

    def measure_train_all(self, pth=None):
        if pth:
            self.metric_model.load_state_dict(torch.load(pth))
        self.measure_model.train()
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.measure_model.parameters(), "lr": 1e-3},
                # {"params": self.metric_model.parameters(), "lr": 1e-3},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.98
        )

        for epoch in tqdm(range(self.epochs)):
            for dataset_name, (type, _) in list(self.dataset[1].items()):
                type, loader = self.get_loader(dataset_name)
                loss = self.measure_train_one_Loader(loader, type)
                self.log(
                    f"{epoch}\t dataset:{dataset_name}\t measure_loss:{loss} \t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.run.log({"epoch": epoch, f"{dataset_name}_measure_loss": loss})
                if epoch % 10 == 0:
                    loss, p, r = self.measure_test(dataset_name)
                    self.run.log({f"{dataset_name}_test_loss": loss, "p": p, "r": r})
                    print(loss, p, r)
            if epoch % 10 == 0:
                torch.save(
                    self.measure_model.state_dict(),
                    f"result/{self.time}/model/measure_{epoch}.pth",
                )
            self.scheduler.step()

    def measure_test(self, dataset_name):
        with torch.no_grad():
            type, loader = self.get_loader(dataset_name, train=False)
            TP = TF = FP = FN = 0
            losses = []
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.reshape(-1)
                _, feature = self.metric_model(x, type)
                pred = self.measure_model(feature)
                loss = self.loss_measure(pred, y)

                gd = torch.eq(y, y.unsqueeze(-1)).float()
                pred = (pred >= 0.5).float()
                TP += torch.sum(pred * gd)
                TF += torch.sum((1 - pred) * (1 - gd))
                FP += torch.sum(pred * (1 - gd))
                FN += torch.sum((1 - pred) * gd)
                losses.append(loss.cpu().item())

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            return (
                sum(losses) / len(losses),
                precision.cpu().item(),
                recall.cpu().item(),
            )

    def eval(self, samples, ys, dataset_name):
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.metrics import adjusted_mutual_info_score

        kmeans = KMeans(len(np.unique(ys)), n_init="auto")
        labels = kmeans.fit_predict(samples)
        # sc = SpectralClustering(len(np.unique(ys)),  n_init=100,
        #                         assign_labels='cluster_qr')
        # labels = sc.fit_predict(samples)
        score = adjusted_mutual_info_score(ys, labels)
        self.log(f"cluster_score:{score}")
        self.run.log({f"{dataset_name}_cluster_score": score})
        print("cluster_score", score)

    def test(self, dataset_name):
        self.metric_model.eval()
        type, loader = self.get_loader(dataset_name, train=False)
        samples = []
        ys = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, pred = self.metric_model(x, type)
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
        self.eval(samples, ys, dataset_name)

        samples_PCA = PCA(n_components=3).fit_transform(samples)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(samples_PCA[:, 0], samples_PCA[:, 1], samples_PCA[:, 2], c=ys)
        plt.show()
        if output:
            plt.savefig(output)
            self.run.log({"dataset_name": dataset_name, "fig": wandb.Image(plt)})
