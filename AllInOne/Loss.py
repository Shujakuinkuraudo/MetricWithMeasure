import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, feature_dim = 1024):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor, y:torch.Tensor): 
        y = y.reshape(-1)
        distance_matrix = torch.cdist(features, features)
        similarity_matrix = nn.functional.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

        equal_sample = torch.eq(y,y.unsqueeze(-1))
        return torch.mean(equal_sample * distance_matrix - ~equal_sample * distance_matrix)  + torch.mean(~equal_sample * similarity_matrix - equal_sample * similarity_matrix) + torch.mean(nn.functional.normalize(features,dim=-1, p=3))
    #     # return torch.sum(equal_sample * torch.acos(similarity_matrix) - ~equal_sample * torch.acos(similarity_matrix))

    # def forward(self, features: torch.Tensor, y:torch.Tensor): 
    #     y = y.reshape(-1)
    #     distance_matrix = torch.cdist(features, features)
    #     distance_matrix = self.softmax(distance_matrix)

    #     similarity_matrix = nn.functional.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    #     similarity_matrix = self.softmax(similarity_matrix)

    #     equal_sample = torch.eq(y,y.unsqueeze(-1))
    #     return torch.sum(equal_sample * distance_matrix)  + torch.sum(~equal_sample * similarity_matrix) + torch.mean(nn.functional.normalize(features, dim=-1))
    #     # return torch.sum(equal_sample * torch.acos(similarity_matrix) - ~equal_sample * torch.acos(similarity_matrix))