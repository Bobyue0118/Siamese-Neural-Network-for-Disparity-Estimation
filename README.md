# Siamese-Neural-Network-for-Disparity-Estimation

1. **Implement the function `sad`**, which given a window size and maximum disparity \( D \), takes a stereo image pair as input and returns a disparity map, computed from the left to the right image. **(10 points)**

2. **Create a visualization** of the computed disparities by implementing the function `visualize_disparity`. It’s a good idea to also visualize the input images to see if the results are sensible. **(10 points)**

3. **Experiment with different window sizes** (for example 3, 7, 15) and report which one leads to better visual results and why? In case you were not able to solve the previous exercises, you can use the provided disparity maps in the `task3/examples/` folder. **(5 points)**

4. **Why do you think the block matching approach** fails to lead to good estimations around homogeneous regions such as the road? **(5 points)**

5. **Develop a Siamese Neural Network architecture** and implement the function `calculate_similarity_score`. For the Siamese Neural Network, you can use the `StereoMatchingNetwork` class. In particular, implement the `__init__` and `forward` methods to initialize the layers and define the forward pass. Details on the architecture can be found in the skeleton code and the reference paper. **(10 points)**

6. **Implement the functions `hinge_loss` and `training_loop`**. After implementation, you can use the script `train.py` to train your Siamese Neural Network. Details about the hinge loss and training process can be found in the reference paper. **(10 points)**

7. **Try to improve the network by finding better hyperparameters**. For example, you can vary the number of training iterations or the number of filters in the convolutional layers. Explain your findings. **(10 points)**

8. **Implement the function `compute_disparity_CNN`** and compare the visualization of the disparity maps from the Siamese Neural Network to the ones obtained by the block matching algorithm. Which predictions are better and why? Can you find regions in the scenes where the differences in predictions are most dominant? (If you were not able to solve the previous exercises, you can use the provided disparity maps in the `task3/examples/` folder.) **(10 points)**
