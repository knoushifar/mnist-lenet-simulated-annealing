# MNIST Classification with LeNet-5 and Simulated Annealing

This project trains a LeNet-5 convolutional neural network on the MNIST handwritten digit dataset. It also includes an optional simulated annealing procedure to search for a better learning rate before final training.

The project was originally developed as a notebook and has been reorganized into a cleaner, reproducible GitHub repository structure.

## Project Highlights

- Implements the LeNet-5 CNN architecture using PyTorch
- Trains and evaluates the model on MNIST digit classification
- Supports automatic MNIST download using `torchvision`
- Supports CSV-based MNIST datasets as an alternative input format
- Includes optional simulated annealing for learning-rate optimization
- Saves the trained model and training-loss plot to the `outputs/` folder

## Repository Structure

```text
mnist-lenet-simulated-annealing/
├── src/
│   └── train.py
├── notebooks/
│   └── AI_original.ipynb
├── data/
│   └── README.md
├── outputs/
│   └── .gitkeep
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mnist-lenet-simulated-annealing.git
cd mnist-lenet-simulated-annealing
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train with automatic MNIST download

```bash
python src/train.py --epochs 5
```

This downloads MNIST through `torchvision`, trains LeNet-5, evaluates it, and saves outputs in the `outputs/` folder.

### Train with simulated annealing learning-rate search

```bash
python src/train.py --optimize-lr --epochs 5 --epochs-per-trial 1
```

This first searches for a learning rate using simulated annealing, then trains the final model using the selected value.

### Train using CSV files

Place the files below inside the `data/` folder:

```text
data/mnist_train.csv
data/mnist_test.csv
```

Then run:

```bash
python src/train.py \
  --data-source csv \
  --train-csv data/mnist_train.csv \
  --test-csv data/mnist_test.csv \
  --epochs 5
```

CSV format must be:

- First column: digit label
- Remaining 784 columns: flattened 28×28 grayscale pixel values

## Outputs

After training, the script saves:

```text
outputs/lenet5_mnist.pt
outputs/training_loss.png
```

If simulated annealing is enabled, it also saves:

```text
outputs/annealing_history.csv
```

## Notes

The uploaded notebook expected both `mnist_train.csv` and `mnist_test.csv`. Since only `mnist_test.csv` was available, the GitHub-ready version uses automatic MNIST download by default. CSV support is still included for reproducibility if both files are available.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Matplotlib

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
