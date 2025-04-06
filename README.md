# Low Cardiac Output Syndrome Predictor

## Description
This project focuses on the identification of prognostic risk factors for Low Cardiac Output Syndrome in 
patients undergoing myocardial revascularization at the Cardiocenter University Hospital Ernesto Che 
Guevara (Villa clara, Cuba) using artificial intelligence (AI) techniques. The main objective of the study is 
to develop a model capable of predicting whether a patient will present the syndrome and, once developed, to 
extract the factors that are most important in this prediction.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [License](#license)

## Installation 
### Prerequisites
- **Git**: To clone this repository, you'll need to have Git installed. You can download it from [git-scm.com](https://git-scm.com/).
- **Anaconda**: Make sure you have Anaconda or Miniconda installed. You can download it from [Anaconda's official website](https://www.anaconda.com/products/distribution#download-section).
- **Python**: This project is compatible with Python 3.12.7. 

### Setup
0. Clone the repository:
   ```bash
   git clone https://github.com/rxhdz/LCOS-Predictor.git
   cd LCOS-Predictor

### Using `environment.yml`
1. Create a new conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the conda environment:
   ```bash
   conda activate low-cardiac-output-syndrome-predictor
   ```

### Using `requirements.txt` (Optional)
If you prefer to create a virtual environment manually or if you want to install additional packages, 
you can use the `requirements.txt` file:
1. Create a new conda environment:
   ```bash
   conda create --name your_environment_name python=3.12.7.
   ```
2. Activate the environment:
   ```bash
   conda activate your_environment_name # Change your_environment_name to your name of preference
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation
To verify that the environment is set up correctly, you can run:
```bash
    conda list  # This will show all installed packages in the environment
```

## Usage
To run the notebooks, open Jupyter Notebook or JupyterLab and navigate to the project directory. You can do so
with the following command:
```bash
    jupyter notebook
```
This should open up your browser, and you should see Jupyter's tree view with the contents of the directory.
Then, open the desired notebook file.

## Notebooks
This project contains the following Jupyter notebooks:

- `01_exploratory_data_analysis.ipynb`: Preprocessing steps, exploratory data analysis and visualizations.
- `02_model_training.ipynb`: Models training and hyperparameter tuning.
- `03_results.ipynb`: Models evaluation, comparison and performance metrics.

## Model Training
You can find the code to train the models in the `02_model_training.ipynb` notebook. It includes:

- Model selection
- Hyperparameter tuning

## Model Evaluation
The evaluation of the models is performed in the `03_results.ipynb` notebook. It includes:

- Models comparison
- Visualizations of models performance
- Visualization of features' importance

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
