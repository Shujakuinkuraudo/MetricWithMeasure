import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor, y: torch.Tensor):
        y = y.reshape(-1)
        distance_matrix = torch.cdist(features, features)
        similarity_matrix = nn.functional.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=-1
        )

        equal_sample = torch.eq(y, y.unsqueeze(-1))
        return torch.mean(
            equal_sample * distance_matrix - ~equal_sample * distance_matrix
        ) + torch.mean(
            ~equal_sample * similarity_matrix - equal_sample * similarity_matrix
        )


class MeasureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.loss = nn.BCELoss()

    def forward(self, digits, y: torch.Tensor):
        y = y.reshape(-1)
        equal_sample = torch.eq(y, y.unsqueeze(-1)).float()
        # print(digits, equal_sample)
        return self.loss(digits * equal_sample, equal_sample) +0.1* self.loss(digits * (1-equal_sample),  equal_sample)
