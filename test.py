import os
import os.path as osp

import numpy as np
import torch
from block_matching import add_padding, visualize_disparity
from dataset import KITTIDataset
from siamese_neural_network import StereoMatchingNetwork


import torch

def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50, patch_size=9):
    """
    Computes the disparity of the stereo image pair.

    Args:
        infer_similarity_metric:  pytorch module object (pre-trained model)
        img_l: tensor holding the left image
        img_r: tensor holding the right image
        max_disparity (int): maximum disparity
        patch_size (int): the size of the patch (assumed to be odd)

    Returns:
        disparity_map: tensor holding the disparity map
    """

    # Set the model to evaluation mode
    infer_similarity_metric.eval()

    # Calculate padding based on patch size
    padding = patch_size // 2

    # Move images and model to the device (e.g., GPU if available)
    device = next(infer_similarity_metric.parameters()).device
    img_l, img_r = img_l.to(device), img_r.to(device)

    # Permute and add batch dimension to match the PyTorch format
    img_l = img_l.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, channels, height, width)
    img_r = img_r.permute(2, 0, 1).unsqueeze(0)

    # Initialize disparity map on the device
    height, width = img_l.shape[2], img_l.shape[3]
    disparity_map = torch.zeros((height, width), device=device)

    # Process in windows across the image to avoid overlapping regions
    window_size = 1

    with torch.no_grad():  # Disable gradients for faster inference
        for y in range(padding, height - padding, window_size):
            for x in range(padding, width - padding, window_size):
                # Initialize variables to store the best disparity and score for each (x, y) position
                max_score = -float('inf')
                best_disparity = 0

                # Extract the left patch once (fixed reference patch)
                left_patch = img_l[:, :, y - padding:y + padding + 1, x - padding:x + padding + 1]
                left_patch = left_patch.permute(0, 2, 3, 1)  # Shape: (1, patch_height, patch_width, channels)

                # Prepare a batch of right patches for each disparity level
                right_patches = []
                for d in range(max_disparity):
                    if x - d - padding < 0:
                        continue
                    right_patch = img_r[:, :, y - padding:y + padding + 1, x - d - padding:x - d + padding + 1]
                    right_patches.append(right_patch)

                # Stack right patches and permute to match the model input
                right_patches = torch.cat(right_patches, dim=0).permute(0, 2, 3, 1)

                # Repeat the left patch to match the batch size of right patches
                left_patches = left_patch.repeat(right_patches.size(0), 1, 1, 1)

                # Compute similarity scores in a single batch
                scores = infer_similarity_metric(left_patches, right_patches).squeeze()

                # Determine the best disparity based on maximum score
                best_disparity = scores.argmax().item()
                disparity_map[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1] = best_disparity

    # Move disparity map back to the CPU
    disparity_map = disparity_map.cpu()

    return disparity_map



def main():
    # Hyperparameters
    training_iterations = 500
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50
    num_of_filters = 64

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network")
    # model_path = osp.join(out_dir, f"trained_model_{training_iterations}_final.pth")
    model_path = osp.join(out_dir, f"siamese_network_itr_{training_iterations}_filters_{num_of_filters}.pth")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Set network to eval mode
    infer_similarity_metric = StereoMatchingNetwork(num_of_filters=num_of_filters)
    infer_similarity_metric.load_state_dict(torch.load(model_path))
    infer_similarity_metric.eval()
    infer_similarity_metric.to("cuda")

    # Load KITTI test split
    dataset = KITTIDataset(osp.join(data_dir, "testing"))
    # Loop over test images
    for i in range(len(dataset)):
        print(f"Processing {i} image")
        # Load images and add padding
        img_left, img_right = dataset[i]
        img_left_padded, img_right_padded = add_padding(img_left, padding), add_padding(
            img_right, padding
        )
        img_left_padded, img_right_padded = torch.Tensor(img_left_padded), torch.Tensor(
            img_right_padded
        )

        # Input to cuda
        img_left_padded, img_right_padded = img_left_padded.to("cuda"), img_right_padded.to("cuda")

        disparity_map = compute_disparity_CNN(
            infer_similarity_metric,
            img_left_padded,
            img_right_padded,
            max_disparity=max_disparity,
        )

        print("Disparity shape:", disparity_map.shape)

        # Visulization
        title = (
            f"Disparity map for image {i} with SNN (training iterations {training_iterations}, "
            f"batch size {batch_size}, patch_size {patch_size})"
        )
        file_name = f"{i}_training_iterations_{training_iterations}_filters{num_of_filters}.png"
        out_file_path = osp.join(out_dir, file_name)
        visualize_disparity(
            disparity_map.squeeze(),
            img_left.squeeze(),
            img_right.squeeze(),
            out_file_path,
            title,
            max_disparity=max_disparity,
        )


if __name__ == "__main__":
    main()
