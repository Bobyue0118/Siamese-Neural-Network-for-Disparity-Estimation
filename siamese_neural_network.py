import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StereoMatchingNetwork(nn.Module):
    """
    The network consists of the following layers:
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - functional.normalize(..., dim=1, p=2)

    Remark: Note that the convolutional layers expect the data to have shape
        `batch_size * channels * height * width`. Permute the input dimensions
        accordingly for the convolutions and remember to revert it before returning the features.
    """

    def __init__(self, num_of_filters=64):
        """
        Implementation of the network architecture.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """
        super(StereoMatchingNetwork, self).__init__()

        # Define convolutional layers for each branch
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_of_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_of_filters, out_channels=num_of_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=num_of_filters, out_channels=num_of_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=num_of_filters, out_channels=num_of_filters, kernel_size=3, padding=1)

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward_once(self, X):
        """
        Passes one of the input patches through the convolutional layers and normalizes the output.

        Args:
            X (torch.Tensor): Image patch of shape (batch_size, height, width, channels)

        Returns:
            features (torch.Tensor): Normalized feature tensor in shape (batch_size, height, width, n_features)
        """
        # Permute input to match (batch_size, channels, height, width)
        X = X.permute(0, 3, 1, 2)

        # Pass through the convolutional layers with ReLU activations
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        X = self.relu(self.conv3(X))
        X = self.conv4(X)

        # Normalize the output along the channel dimension (dim=1)
        X = F.normalize(X, p=2, dim=1)

        # Permute back to (batch_size, height, width, channels)
        X = X.permute(0, 2, 3, 1)

        return X

    def forward(self, Xl, Xr):
        """
        The forward pass for two input patches to compute a similarity score.

        Args:
            Xl (torch.Tensor): Left image patch of shape (batch_size, height, width, channels)
            Xr (torch.Tensor): Right image patch of shape (batch_size, height, width, channels)

        Returns:
            score (torch.Tensor): Similarity score between the left and right patches
        """
        # Get normalized features for both the left and right patches
        features_l = self.forward_once(Xl)
        features_r = self.forward_once(Xr)

        # Compute the dot product similarity score
        # Since features are now (batch_size, height, width, n_features), permute them back to (batch_size, n_features, height, width)
        features_l = features_l.permute(0, 3, 1, 2)
        features_r = features_r.permute(0, 3, 1, 2)

        # Perform element-wise multiplication and sum across spatial and channel dimensions
        score = torch.sum(features_l * features_r, dim=(1,2,3))/(torch.norm(features_l, p=2, dim=(1, 2, 3))*torch.norm(features_r, p=2, dim=(1, 2, 3)))
        # print('score', score)
        return score


def calculate_similarity_score(infer_similarity_metric, Xl, Xr, patch_size=9):
    """
    Computes the similarity score for two stereo image patches.

    Args:
        infer_similarity_metric (torch.nn.Module):  pytorch module object
        Xl (torch.Tensor): tensor holding the left image patch
        Xr (torch.Tensor): tensor holding the right image patch
        patch_size (int): size of the image patch

    Returns:
        score (torch.Tensor): the similarity score of both image patches which is the dot product of their features
    """

    # Pass both patches through the model to get the similarity score
    score = infer_similarity_metric(Xl, Xr)

    return score
