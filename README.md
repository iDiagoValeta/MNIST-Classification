# MNIST-Classification

**A two-layer neural network written from scratch in NumPy that classifies handwritten digits from the Kaggle "Digit Recognizer" (MNIST) data, trained with Nesterov momentum.**

> **Context.** Built in April 2025 as a personal learning project, when I was a Computer Engineering student at UPV and had just started publishing on GitHub. It is one of my first public repositories and a first hands-on exercise in how neural networks learn. It is kept as it was first published; the code is not maintained.

## What it does

- Loads the Kaggle Digit Recognizer `train.csv` (28x28 grayscale digits, flattened to 784 pixels per row).
- Shuffles the data, holds out 1,000 rows as a test split and scales pixels to the range 0 to 1.
- Trains a 784 - 10 - 10 network (ReLU hidden layer, softmax output) using only NumPy, with hand-written forward and backward passes.
- Replaces plain gradient descent with Nesterov momentum (`mu = 0.9`, learning rate 0.10, 500 iterations).
- Shows predictions against labels for ten images, plotted with matplotlib.

## How it works

The model is adapted from a well-known from-scratch MNIST network credited in the notebook to "Samsom Zhang" (spelled that way there). The changes are a rescaled parameter initialisation (`init_params`) and the replacement of the gradient descent step with Nesterov momentum. At each iteration the weights are moved to a look-ahead point (`W + mu * v`), the gradients are computed there, the velocities are updated (`v = mu * v - alpha * dW`) and the real step is applied. Training is full batch: every iteration uses the whole training matrix.

## Results

These numbers are the printed outputs saved in `mnist_Classification.ipynb`:

- Training accuracy rises from 0.100 at iteration 0 to 0.937 at iteration 490.
- The final evaluation cell reports 0.937 accuracy (`0.9371707...`), but it calls `make_predictions(X_train, ...)`, so it measures the training set. The 1,000-row held-out split (`X_test`, `Y_test`) is created but never scored in the notebook, so no held-out accuracy is recorded. The notebook text claims "over 90%" on test data; that claim is not backed by the saved outputs.
- In the ten sample predictions shown, nine match their label and one does not (predicted 1, label 4).

## Run it

Not re-run for this README; the steps below come from reading the notebook.

1. Unzip `digit-recognizer.zip` to get `train.csv` (also contains `test.csv` and `sample_submission.csv`, unused by the notebook).
2. Open `mnist_Classification.ipynb`. It was written for Google Colab: the first code cell mounts Google Drive and reads `/content/drive/MyDrive/Colab Notebooks/mnist/train.csv`. To run it elsewhere, remove the `google.colab` lines and change the path to your local `train.csv`.
3. Needs Python 3 with `numpy`, `pandas` and `matplotlib`. No GPU is required.
4. Run all cells in order.

## Repository layout

```
mnist_Classification.ipynb   the full notebook: data prep, network, Nesterov training, predictions
digit-recognizer.zip         Kaggle Digit Recognizer data (train.csv, test.csv, sample_submission.csv)
```

## Stack

Python, NumPy, pandas, matplotlib, Jupyter / Google Colab.
