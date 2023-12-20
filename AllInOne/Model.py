import torch
import torch.nn as nn


# image_feature
class ImageFeatureExtraction(torch.nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), "same"),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64)),
        )
        self.Linear = nn.LazyLinear(output_shape)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.Linear(x.flatten())
        return x


class TextFeatureExtraction(torch.nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.embedding = nn.Embedding(10000, 256)
        self.LSTM = nn.LSTM(256, 512, 2, batch_first=True)
        self.Linear = nn.LazyLinear(output_shape)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.LSTM(x)
        x = self.Linear(x.flatten())
        return x


class AudioFeatureExtraction(torch.nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()

        self.Linear = nn.LazyLinear(output_shape)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.Linear(x.flatten())
        return x


class AutoEncoder(nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(output_shape // 4),
            nn.ReLU(),
            nn.LazyLinear(output_shape // 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(output_shape // 4), nn.ReLU(), nn.LazyLinear(output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Model(nn.Module):
    def __init__(self, output_shape=1024):
        super().__init__()
        self.FeatureExtraction = nn.ModuleDict(
            {
                "Image": ImageFeatureExtraction(output_shape),
                "Text": TextFeatureExtraction(output_shape),
                "Audio": AudioFeatureExtraction(output_shape),
            }
        )
        self.ae = AutoEncoder(output_shape)

    def forward(self, x: torch.Tensor, type):
        x = self.FeatureExtraction[type](x)
        x_ae = self.ae(x)
        return x, x_ae
