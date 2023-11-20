import torch
import torch.nn as nn

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

        return torch.sum(distancias)