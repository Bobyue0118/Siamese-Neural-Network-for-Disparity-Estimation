import os
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
from dataset import KITTIDataset, PatchProvider
from siamese_neural_network import StereoMatchingNetwork, calculate_similarity_score

import matplotlib.pyplot as plt
from itertools import islice


def hinge_loss(score_pos, score_neg, margin=0.2):
    """
    Computes the hinge loss for the similarity of a positive and a negative example.
    """
    loss = torch.clamp(margin + score_neg - score_pos, min=0)
    avg_loss = loss.mean()  # Average loss over the batch
    acc = (score_pos > score_neg).float().mean()

    return avg_loss, acc


def training_loop(
    infer_similarity_metric,
    patches,
    optimizer,
    out_dir,
    device,
    iterations=1000,
    batch_size=128,
    num_of_filters=64,
):
    """
    Runs the training loop of the siamese network.
    """
    infer_similarity_metric.train()
    training_losses = []  # List to store loss values for each iteration

    # Limit to `iterations` batches
    limited_batches = islice(patches.iterate_batches(batch_size), iterations)

    with tqdm(enumerate(limited_batches), total=iterations, desc="Training Iterations") as pbar:
        for i, (ref_batch, pos_batch, neg_batch) in pbar:

            # Concatenate positive and negative pairs
            left_batch = torch.cat((ref_batch, ref_batch), dim=0)
            right_batch = torch.cat((pos_batch, neg_batch), dim=0)

            # Move data to GPU
            left_batch = left_batch.to(device)
            right_batch = right_batch.to(device)

            # Calculate similarity scores
            score_pos = calculate_similarity_score(infer_similarity_metric, left_batch[:batch_size], right_batch[:batch_size])
            score_neg = calculate_similarity_score(infer_similarity_metric, left_batch[batch_size:], right_batch[batch_size:])

            # Calculate hinge loss and accuracy
            loss, acc = hinge_loss(score_pos, score_neg)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store the loss value for plotting
            training_losses.append(loss.item())

            # Print error and accuracy each iteration
            pbar.set_postfix({"Loss": loss.item(), "Accuracy": acc.item()})

            # Save model checkpoint and plot training loss curve at specified intervals
            if i in [iterations // 5-1, iterations // 2-1, iterations-1]:
                # Save model checkpoint
                model_path = osp.join(out_dir, f"siamese_network_itr_{i+1}_filters_{num_of_filters}.pth")
                torch.save(infer_similarity_metric.state_dict(), model_path)
                print(f"Model saved to {model_path}")

                # Plot and save the training loss curve
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.legend()
                plt.grid()

                # Save the plot as an image file
                plot_path = osp.join(out_dir, f"training_loss_curve_{i+1}_filters_{num_of_filters}.png")
                plt.savefig(plot_path)
                plt.close()  # Free up resources after saving the plot
                print(f"Training loss curve saved to {plot_path}")
            if i >= iterations-1:
                break



def main():
    np.random.seed(7)
    torch.manual_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_iterations = 1000          # 200, 500, 1000
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9
    num_of_filters = 32                # 16, 32, 64
    padding = patch_size // 2
    max_disparity = 50

    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    dataset = KITTIDataset(
        osp.join(data_dir, "training"),
        osp.join(data_dir, "training/disp_noc_0"),
    )
    patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))

    infer_similarity_metric = StereoMatchingNetwork(num_of_filters=num_of_filters).to(device)
    infer_similarity_metric.train()

    optimizer = torch.optim.SGD(
        infer_similarity_metric.parameters(), lr=learning_rate, momentum=0.9
    )

    print(f"Iterations: {training_iterations}, filters: {num_of_filters}")

    training_loop(
        infer_similarity_metric,
        patches,
        optimizer,
        out_dir,
        device,
        iterations=training_iterations,
        batch_size=batch_size,
        num_of_filters=num_of_filters
    )

    patches.stop()


if __name__ == "__main__":
    main()
