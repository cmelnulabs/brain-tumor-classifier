Brain Tumor Classifier
======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Overview
--------
This project implements a convolutional neural network (CNN) for multi-class classification of brain tumor MRI images. It uses the public Kaggle dataset `masoudnickparvar/brain-tumor-mri-dataset` and applies class balancing techniques.

Sections
--------
.. contents::
   :local:

1. Dataset Acquisition
----------------------
- Downloads dataset from KaggleHub:
  - masoudnickparvar/brain-tumor-mri-dataset
- Classes are auto-detected from directory names.
- Training and testing sets are used as provided.

2. Data Preprocessing
---------------------
- Converts images to grayscale.
- Resizes images to 224x224 pixels.
- Normalizes pixel values to mean 0.5, std 0.5.

3. Class Balancing
------------------
- Calculates class weights inversely proportional to class frequency.
- Applies class weights in CrossEntropyLoss for balanced learning.

4. Model Architecture
---------------------
- Sequential CNN with:
  - Seven convolutional layers (1→8→16→32→64→128→256→512 channels)
  - Batch normalization after each convolution
  - ReLU activation
  - Max pooling after each convolution
  - Dropout (p=0.2) after the 64-channel block
  - Flatten layer
  - Fully connected layer to output number of classes

5. Training Procedure
---------------------
- Trains for up to 15 epochs with early stopping (patience=3).
- Tracks training loss and accuracy.
- Evaluates on test set after each epoch.
- Saves the model with the highest test accuracy.

6. Evaluation
-------------
- Loads and evaluates the best saved model on the test set.
- Reports test loss and test accuracy.

7. Usage
--------
- Run `python -m main` to train and evaluate the model.
- Prints image counts per class, training progress, and final results.
- Hyperparameters can be adjusted in `btclassifier/config.py`.

8. Extending the Model
----------------------
- You can adjust image size, batch size, number of layers, dropout, and augmentation.
- Modular codebase: add new features in the `btclassifier/` package.


9. Implementation Decisions for Multiclass Classification
---------------------------------------------------------
This project is designed for multiclass classification, where each MRI image belongs to one of several tumor types. Key implementation choices include:

- **Output Layer:** The final layer uses a fully connected (linear) layer with output size equal to the number of classes, followed by softmax activation (via CrossEntropyLoss).
- **Loss Function:** CrossEntropyLoss is used, which is standard for multiclass problems and combines softmax and negative log-likelihood in one step.
- **Class Weights:** To address class imbalance, class weights are computed inversely proportional to class frequency and passed to the loss function. This ensures minority classes are not ignored during training.
- **Evaluation Metrics:** Accuracy is reported, but the modular codebase allows for easy addition of per-class precision, recall, or confusion matrix analysis.
- **Data Structure:** The dataset is organized in folders named after each class, allowing automatic class detection and label assignment.
- **No One-vs-All:** The model is trained directly for multiclass (not one-vs-all or binary) classification, leveraging PyTorch's efficient handling of multiclass targets.

These choices ensure the model is robust, fair to all classes, and easy to extend for more classes or different metrics.
