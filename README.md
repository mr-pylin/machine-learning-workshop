# 🤖 Machine Learning Workshop

[![License](https://img.shields.io/github/license/mr-pylin/machine-learning-workshop?color=blue)](https://github.com/mr-pylin/machine-learning-workshop/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.13.1-yellow?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3131/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3e38c99416c9400facb62a4349ea8802)](https://app.codacy.com/gh/mr-pylin/machine-learning-workshop/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![Repo Size](https://img.shields.io/github/repo-size/mr-pylin/machine-learning-workshop?color=lightblue)
![Last Updated](https://img.shields.io/github/last-commit/mr-pylin/machine-learning-workshop?color=orange)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?color=brightgreen)](https://github.com/mr-pylin/machine-learning-workshop/pulls)

A guide to machine learning, covering core concepts, algorithms, and practical applications using **scikit-learn**.

## 📖 Table of Contents

1. [**Introduction**](./)
<!-- What is Machine Learning?
Types of ML: Supervised, Unsupervised, Reinforcement
Typical ML Pipeline
Tools and Libraries Overview (NumPy, pandas, scikit-learn, PyTorch/TensorFlow) -->
1. [**Math for Machine Learning**](./)
<!-- Linear Algebra Basics
Probability & Statistics
Calculus (Gradients, Derivatives)
Optimization (Gradient Descent) -->
1. [**Data Handling**](./)
<!-- Data Collection & Loading
Data Cleaning & Preprocessing
Feature Engineering
Splitting Data (Train/Val/Test) -->
1. [**Supervised Learning**](./)
<!-- Regression
Linear Regression
Polynomial Regression
Classification
Logistic Regression
k-NN
Decision Trees & Random Forests
Support Vector Machines (SVM)
Naive Bayes -->
1. [**Model Evaluation**](./)
<!-- Metrics for Regression (MSE, RMSE, R²)
Metrics for Classification (Accuracy, Precision, Recall, F1, ROC-AUC)
Cross-Validation
Bias-Variance Tradeoff -->
1. [**Unsupervised Learning**](./)
<!-- Clustering: k-Means, DBSCAN, Hierarchical
Dimensionality Reduction: PCA, t-SNE, UMAP
Association Rules -->
1. [**Model Tuning & Selection**](./)
<!-- Hyperparameter Tuning (Grid Search, Random Search, Bayesian Optimization)
Regularization (L1, L2)
Pipelines -->
1. [**Neural Networks & Deep Learning**](./)
<!-- Perceptron & MLP
Backpropagation
Activation Functions
CNNs for Images
RNNs/LSTMs for Sequences
Transfer Learning -->
1. [**Production & Deployment**](./)
<!-- Saving/Loading Models
Model Inference API (FastAPI, Flask)
Model Monitoring & Drift
Introduction to MLOps -->
1. [**Bonus Topics**](./)
<!-- Feature Selection & Importance
Explainability (SHAP, LIME)
Ethics in ML
AutoML
Ensemble Learning (Bagging, Boosting) -->

## 📋 Prerequisites

- 👨‍💻 **Programming Fundamentals**
  - Proficiency in **Python** (data types, control structures, functions, classes, etc.).
    - My Python Workshop: [github.com/mr-pylin/python-workshop](https://github.com/mr-pylin/python-workshop)
  - Experience with libraries like **NumPy**, **Pandas** and **Matplotlib**.
    - My NumPy Workshop: [github.com/mr-pylin/numpy-workshop](https://github.com/mr-pylin/numpy-workshop)
    - My Pandas Workshop: [Coming Soon](https://github.com/mr-pylin/#)
    - My Data Visualization Workshop: [github.com/mr-pylin/data-visualization-workshop](https://github.com/mr-pylin/data-visualization-workshop)
- 🔣 **Mathematics for Machine Learning**
  - 🔲 **Linear Algebra**: Vectors, matrices, matrix operations.
    - [**Linear Algebra Review and Reference**](https://www.cs.cmu.edu/%7Ezkolter/course/linalg/linalg_notes.pdf) written by [*Zico Kolter*](https://zicokolter.com).
    - [**Notes on Linear Algebra**](https://webspace.maths.qmul.ac.uk/p.j.cameron/notes/linalg.pdf) written by [*Peter J. Cameron*](https://cameroncounts.github.io/web).
    - [**MATH 233 - Linear Algebra I Lecture Notes**](https://www.geneseo.edu/~aguilar/public/assets/courses/233/main_notes.pdf) written by [*Cesar O. Aguilar*](https://www.geneseo.edu/~aguilar/).
  - 📈 **Calculus**: Derivatives, gradients, partial derivatives, chain rule (for backpropagation).
    - [**Lecture notes on advanced gradient descent**](https://www.lamsade.dauphine.fr/~croyer/ensdocs/GD/LectureNotesOML-GD.pdf) written by [*Cl´ement W. Royer*](https://scholar.google.fr/citations?user=nmRlYWwAAAAJ&hl=en).
    - [**MATH 221 –  CALCULUS LECTURE NOTES VERSION 2.0**](https://people.math.wisc.edu/~angenent/Free-Lecture-Notes/free221.pdf) written by [*Sigurd Angenent*](https://people.math.wisc.edu/~angenent).
    - [**Calculus**](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) written by [*Gilbert Strang*](https://math.mit.edu/~gs).
  - 🎲 **Probability & Statistics**: Probability distributions, mean/variance, etc.
    - [**MATH1024: Introduction to Probability and Statistics**](https://www.sujitsahu.com/teach/2020_math1024.pdf) written by [*Sujit Sahu*](https://www.southampton.ac.uk/people/5wynjr/professor-sujit-sahu).

## ⚙️ Setup

This project requires Python **v3.10** or higher. It was developed and tested using Python **v3.13.1**. If you encounter issues running the specified version of dependencies, consider using this version of Python.

### 📝 List of Dependencies

[![ipykernel](https://img.shields.io/badge/ipykernel-6.30.0-ff69b4)](https://pypi.org/project/ipykernel/6.30.0/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-8.1.7-ff6347)](https://pypi.org/project/ipywidgets/8.1.7/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.3-green)](https://pypi.org/project/matplotlib/3.10.3/)
[![numpy](https://img.shields.io/badge/numpy-2.3.2-orange)](https://pypi.org/project/numpy/2.3.2/)
[![pandas](https://img.shields.io/badge/pandas-2.3.1-yellow)](https://pypi.org/project/pandas/2.3.1/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-darkblue)](https://pypi.org/project/scikit-learn/1.7.1/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13.2-lightblue)](https://pypi.org/project/seaborn/0.13.2/)

### 📦 Install Dependencies

#### 📦 Method 1: Poetry (**Recommended** ✅)

Use [**Poetry**](https://python-poetry.org/) for dependency management. It handles dependencies, virtual environments, and locking versions more efficiently than pip.  
To install exact dependency versions specified in [**poetry.lock**](./poetry.lock) for consistent environments **without** installing the current project as a package:

```bash
poetry install --no-root
```

#### 📦 Method 2: Pip

Install all dependencies listed in [**requirements.txt**](./requirements.txt) using [**pip**](https://pip.pypa.io/en/stable/installation/):

```bash
pip install -r requirements.txt
```

### 🛠️ Usage Instructions

1. Open the root folder with [**VS Code**](https://code.visualstudio.com/) (`Ctrl/Cmd + K` followed by `Ctrl/Cmd + O`).
1. Open `.ipynb` files using the [**Jupyter extension**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) integrated with **VS Code**.
1. Select the correct Python kernel and virtual environment where the dependencies were installed.
1. Allow **VS Code** to install any recommended dependencies for working with Jupyter Notebooks.

✍️ **Notes**:  

- It is **highly recommended** to stick with the exact dependency versions specified in [**poetry.lock**](./poetry.lock) or [**requirements.txt**](./requirements.txt) rather than using the latest package versions. The repository has been **tested** on these versions to ensure **compatibility** and **stability**.
- This repository is **actively maintained**, and dependencies are **updated regularly** to the latest **stable** versions.
- The **table of contents** embedded in the **notebooks** may not function correctly on **GitHub**.
- For an improved experience, open the notebooks **locally** or view them via [**nbviewer**](https://nbviewer.org/github/mr-pylin/mavhine-learning-workshop).

## 🔗 Useful Links

### **scikit-learn**

- **Source Code**:
  - Over **3000** contributors are currently working on scikit-learn.
  - *Link*: [github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
- **Website**:
  - The **official** website for scikit-learn, providing user guides, examples, documentation, and installation instructions.
  - Link: [scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **Stable Documentations**:
  - Comprehensive reference covering all core modules, estimators, and utilities. Also includes tutorials and examples to get started with machine learning.
  - Link: [scikit-learn.org/stable/index.html](https://scikit-learn.org/stable/index.html)
- **Tutorials & Examples Gallery**:
  - A curated collection of end-to-end examples, from basic model fitting to advanced pipelines and visualizations.
  - Link: [scikit-learn.org/stable/auto_examples/index.html](https://scikit-learn.org/stable/auto_examples/index.html)
- **User Guide**:
  - Step-by-step explanations of scikit-learn concepts and usage patterns, including preprocessing, model selection, and evaluation.
  - Link: [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)

### **NumPy**

- A fundamental package for scientific computing in Python, providing support for **arrays**, **matrices**, and a large collection of **mathematical functions**.
- Official site: [numpy.org](https://numpy.org/)

### **Pandas**

- A powerful, open-source data analysis and manipulation library for Python.
- Pandas is built on top of NumPy.
- Official site: [pandas.pydata.org](https://pandas.pydata.org/)

### **Data Visualization**

- A comprehensive collection of Python libraries for creating static, animated, and interactive visualizations: **Matplotlib**, **Seaborn**, and **Plotly**.
- Official sites: [matplotlib.org](https://matplotlib.org/) | [seaborn.pydata.org](https://seaborn.pydata.org/) | [plotly.com](https://plotly.com/)

## 🔍 Find Me

Any mistakes, suggestions, or contributions? Feel free to reach out to me at:

- 📍[**linktr.ee/mr_pylin**](https://linktr.ee/mr_pylin)

I look forward to connecting with you! 🏃‍♂️

## 📄 License

This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to **use**, **modify**, and **distribute** this code, but you **must** include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.

### ©️ Copyright Information

- **Original Images**:
  - The images located in the [./assets/images/original/](./assets/images/original/) folder are licensed under the **[CC BY-ND 4.0](./assets/images/original/LICENSE)**.
  - Note: This license restricts derivative works, meaning you may share these images but cannot modify them.
- **Third-Party Assets**:
  - Additional images located in [./assets/images/third_party/](./assets/images/third_party/) are used with permission or according to their original licenses.
  - Attributions and references to original sources are included in the code where these images are used.
