import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, feature_dim = 1024):
        super().__init__()

    def forward(self, features: torch.Tensor, y:torch.Tensor): 
        y = y.reshape(-1)
        distance_matrix = torch.cdist(features, features)
        similarity_matrix = nn.functional.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

        equal_sample = torch.eq(y,y.unsqueeze(-1))
        return torch.sum(equal_sample * distance_matrix - ~equal_sample * distance_matrix)  + torch.sum(~equal_sample * similarity_matrix - equal_sample * similarity_matrix)
        # return torch.sum(equal_sample * torch.acos(similarity_matrix) - ~equal_sample * torch.acos(similarity_matrix))