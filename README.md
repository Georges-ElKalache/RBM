# RBM Image Generation Project

This project implements a Restricted Boltzmann Machine (RBM) to learn and generate images from datasets such as Binary AlphaDigits and MNIST. The project explores various aspects of RBM training, including hyperparameter tuning, cross-analyses, and alternative models.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/RBM_Image_Generation.git
    cd RBM_Image_Generation
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Project Structure

- `RBM.ipynb`: Main notebook containing the implementation of the RBM model, training, and image generation.
- `alternative_models.ipynb`: Notebook exploring alternative models and comparisons with RBM.
- `analyses_croisées.ipynb`: Notebook for cross-analyses and evaluation of model performance.
- `hyperparameters.ipynb`: Notebook dedicated to hyperparameter tuning and optimization.
- `characters.ipynb`: Notebook dedicated to character number influence testing.
- `mnist.ipynb`: Notebook for training and evaluating the RBM on the MNIST dataset.
- `data/`: Directory to store datasets (e.g., Binary AlphaDigits, MNIST).
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Project documentation.

## Usage

### Running the Notebooks
Each notebook is designed to be self-contained and can be run independently. Here’s a brief overview of what each notebook does:

1. **RBM.ipynb**:
   - Implements the RBM model.
   - Trains the model on the Binary AlphaDigits dataset.
   - Generates and visualizes images learned by the RBM.

2. **alternative_models.ipynb**:
   - Explores alternative models (e.g., Autoencoders, GANs) and compares their performance with RBM.

3. **analyses_croisées.ipynb**:
   - Performs cross-analyses to evaluate the impact of different parameters and datasets on model performance.

4. **hyperparameters.ipynb**:
   - Focuses on hyperparameter tuning (e.g., learning rate, number of hidden units) to optimize the RBM.

5. **characters.ipynb**:
   - Focuses on number of characters tuning to understand influence on the RBM performance.

6. **mnist.ipynb**:
   - Trains the RBM on the MNIST dataset.
   - Evaluates the model's ability to generate handwritten digits.

### Example: Training the RBM
To train the RBM on the Binary AlphaDigits dataset, open the `RBM.ipynb` notebook and follow the steps:

1. Load the dataset:
    ```python
    from utils import load_binary_alpha_digits
    data = load_binary_alpha_digits()
    ```

2. Initialize and train the RBM:
    ```python
    from RBM_model import RBM
    rbm = RBM(n_visible=data.shape[1], n_hidden=64)
    rbm.train(data, n_epochs=50, learning_rate=0.01, batch_size=10)
    ```

3. Generate and visualize images:
    ```python
    from generate import generate_images
    generated_images =
