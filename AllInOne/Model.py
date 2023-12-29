import torch
import torch.nn as nn


# image_feature
class ImageFeatureExtraction(torch.nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding="same"),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64)),
        )
        self.Linear = nn.LazyLinear(output_shape)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        return x


class TextFeatureExtraction(torch.nn.Module):
    def __init__(self, output_shape=5):
        super().__init__()
        # self.embedding = nn.Embedding(50265, 256)
        self.embedding = nn.Embedding(100000, 256)
        self.LSTM = nn.LSTM(256, 128, 2, batch_first=True)
        self.Linear = nn.LazyLinear(output_shape)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        (x, (hn, cn)) = self.LSTM(x)
        x = self.Linear(x.reshape(x.size(0), -1))
        return x


class AudioFeatureExtraction(torch.nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding="same"),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64)),
        )
        self.Linear = nn.LazyLinear(output_shape)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.Linear(x.view(x.size(0), -1))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(output_shape // 4),
            nn.ReLU(),
            nn.LazyLinear(output_shape // 16),
            nn.ReLU(),
            nn.LazyLinear(output_shape // 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(output_shape // 16),
            nn.ReLU(),
            nn.LazyLinear(output_shape // 4),
            nn.ReLU(),
            nn.LazyLinear(output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MetricModel(nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.FeatureExtraction = nn.ModuleDict(
            {
                # "Image1": ImageFeatureExtraction(200),
                # "Image2": ImageFeatureExtraction(200),
                "Text": TextFeatureExtraction(output_shape),
                # "Audio": AudioFeatureExtraction(200),
            }
        )
        self.ae = AutoEncoder(output_shape)

    def forward(self, x: torch.Tensor, type):
        x = self.FeatureExtraction[type](x)
        x = nn.functional.tanh(x)
        # x = x / torch.sum(nn.functional.normalize(x))
        
        x_ae = self.ae(x)
        x_ae = nn.functional.tanh(x_ae)
        # x_ae = x_ae / torch.sum(nn.functional.normalize(x_ae))
        return x, x_ae


class MeasureModel(nn.Module):
    def __init__(self, latnet_shape=256):
        super().__init__()
        self.Linears = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
        )
        self.L = nn.LazyLinear(latnet_shape)
        self.Mid = nn.Linear(latnet_shape,latnet_shape,bias=False)
        self.R = nn.LazyLinear(latnet_shape)
        self.ac = nn.Sigmoid()
        self.sf = nn.Softmax(dim=-1)
        self.classify = nn.LazyLinear(5)

    def forward(self, x: torch.tensor): # x: [batch_size, 1024]
        x_ = self.sf(x/0.1)
        # return self.ac(self.L(x) @ self.Mid.weight @ self.R(x).T)
        # return self.ac(self.L(x) @ self.R(x).T)
        return x_ @ x_.T, x
        
