# Data

The training script downloads MNIST automatically through `torchvision` by default.

If you want to use CSV files instead, place them here using this format:

- `mnist_train.csv`
- `mnist_test.csv`

Each CSV should contain the digit label in the first column and 784 pixel values in the remaining columns.
