# X-ray Chest Classification

## Project Description

This project implements a deep learning model for chest X-ray image classification. The model is trained to classify chest X-ray images into three categories:

- **Normal**: Healthy chest X-rays
- **COVID**: Chest X-rays showing signs of COVID-19
- **Pneumonia**: Chest X-rays showing signs of pneumonia

## Dataset

The dataset used in this project is sourced from Kaggle:

**Dataset Link**: [COVID Pneumonia Normal Chest X-ray Images](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images)

The dataset contains chest X-ray images organized into three classes (COVID, Normal, and Pneumonia) with a substantial number of images for training and evaluation.

## Project Structure

- `main.ipynb` - Main Jupyter notebook containing data exploration, model training, and evaluation
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Installation

1. Clone or download this project
2. Set up virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset from the Kaggle link above
2. Open `main.ipynb` in Jupyter Notebook
3. Follow the notebook cells to:
   - Load and explore the dataset
   - Preprocess the images
   - Train the classification model
   - Evaluate model performance

## Requirements

See `requirements.txt` for all dependencies and their versions.

## License

This project is for educational purposes.

