import torch
import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self, option, weight_lambda):
        super(JointLoss, self).__init__()
        assert weight_lambda >= 0 # Raise exception if weight_lambda is negative
        self.weight_lambda = weight_lambda
        if option == 'mse':
            self.loss = nn.MSELoss()
        if option == 'huber':
            self.loss = nn.HuberLoss()

    def forward(self, input1, target1, input2 = None, target2 = None):
        # Calculate MSE loss for the imitation network
        imit_loss = self.loss(input1, target1)
        
        # Calculate MSE loss for the prediction network if weight lambda is positive
        pred_loss = None
        if self.weight_lambda > 0:
            pred_loss = self.loss(input2, target2)
        
        # Combine the losses with the specified weight lambda
        if pred_loss is None:
            joint_loss = imit_loss
        else:
            joint_loss = imit_loss + self.weight_lambda * pred_loss
        
        return joint_loss
