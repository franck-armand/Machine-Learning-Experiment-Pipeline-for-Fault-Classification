## Machine Learning Experiment Pipeline for Fault Classification

This repository contains code for running various machine learning experiments on a dataset, with a focus on fault classification. The pipeline performs the following tasks:

- **Preprocessing**: Data scaling and feature selection
- **Model Training**: Several machine learning models including Random Forest, SVM, Naive Bayes, CNN, and ensemble methods (Stacking, Voting, and Bagging classifiers)
- **Evaluation**: Performance evaluation using accuracy score
- **Visualization**: Results visualization and model saving
- **Logging**: Logging the training process, results, and models

#### Features

- **Fault Type Mapping**: Maps numerical fault types to descriptive labels for easier understanding.
- **Custom Feature Selection**: Feature selection using Mutual Information (MI) and Recursive Feature Elimination (RFE).
- **Multiple Models**: The pipeline supports multiple models like Random Forest, SVM, Naive Bayes, and CNN.
- **Ensemble Learning**: Includes methods like Stacking, Voting, and Bagging classifiers.
- **Logging & Saving**: Logs results and training logs to CSV and saves models.

#### Requirements

The following Python libraries are required to run the code:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `os`
- `datetime`

You can install the necessary dependencies with:

```bash
pip install -r requirements.txt
```