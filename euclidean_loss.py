import torch
import torch.nn as nn

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, output, target):
        # Reshape outputs and targets to (T * 54, 2)
        output = output.view(-1, 2)
        target = target.view(-1, 2)
        
        # Compute the squared Euclidean distance
        distancias = torch.sum((output - target) ** 2, dim=1)
        
        # Sum of squared distances
        return torch.mean(distancias)
    
class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, output, target):
        
        output_x = output[:, :, 0].reshape(-1, 1)
        output_y = output[:, :, 1].reshape(-1, 1)
        
        target_x = target[:, :, 0].reshape(-1, 1)
        target_y = target[:, :, 1].reshape(-1, 1)
        

        _output = torch.stack([output_x, output_y], dim=1)
        _target = torch.stack([target_x, target_y], dim=1)

        distancias = torch.norm(_output - _target, dim=1)
        # distancias = torch.sum(torch.abs(_output - _target), dim=1) # Manhattan distance
        return torch.sum(distancias)
    
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        loss = self.weights * (output - target) ** 2
        return loss.mean()


'''
class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, output, target):
        
        output_x = output[:, :, 0].reshape(-1, 1)
        output_y = output[:, :, 1].reshape(-1, 1)
        
        target_x = target[:, :, 0].reshape(-1, 1)
        target_y = target[:, :, 1].reshape(-1, 1)
        

        _output = torch.stack([output_x, output_y], dim=1)
        _target = torch.stack([target_x, target_y], dim=1)

        distancias = torch.norm(_output - _target, dim=1)
        # distancias = torch.sum(torch.abs(_output - _target), dim=1) # Manhattan distance
        return torch.sum(distancias)
'''