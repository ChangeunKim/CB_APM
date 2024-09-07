import torch
import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self, weight_lambda):
        super(JointLoss, self).__init__()
        assert weight_lambda >= 0 # Raise exception if weight_lambda is negative
        self.weight_lambda = weight_lambda
        self.loss = nn.MSELoss()

    def forward(self, input1, target1, input2 = None, target2 = None):
        # Calculate MSE loss for the prediction network
        pred_loss = self.loss(input1, target1)
        
        # Calculate MSE loss for the consensus network if weight lambda is positive
        imit_loss = 0
        if self.weight_lambda > 0:
            imit_loss = self.loss(input2, target2)
        
        # Combine the losses with the specified weight lambda
        joint_loss = pred_loss + self.weight_lambda * imit_loss
        
        return joint_loss
