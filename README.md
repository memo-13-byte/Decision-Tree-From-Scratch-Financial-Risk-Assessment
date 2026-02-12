# 🌳 Decision Tree from Scratch - Financial Risk Assessment

A comprehensive implementation of the ID3 Decision Tree algorithm from scratch for financial risk assessment, featuring custom entropy calculations, information gain optimization, and detailed data preprocessing.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.2-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Highlights](#technical-highlights)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a **Decision Tree classifier using the ID3 algorithm** entirely from scratch without using scikit-learn's DecisionTreeClassifier. The implementation focuses on financial risk assessment, predicting loan default risk based on various customer financial attributes.

The project demonstrates deep understanding of:
- Information theory (entropy and information gain)
- Tree-based learning algorithms
- Data preprocessing and feature engineering
- Handling missing values with statistical methods
- Custom algorithm implementation without ML libraries

## ✨ Features

### Core Implementation
- ✅ **Custom ID3 Algorithm**: Complete decision tree implementation from scratch
- ✅ **Entropy Calculation**: Mathematical foundation for information theory
- ✅ **Information Gain**: Optimal feature selection at each node
- ✅ **Tree Visualization**: Human-readable tree structure output
- ✅ **Prediction Engine**: Efficient classification for new samples

### Data Processing
- 🔧 **Median Imputation**: Statistical approach for numerical missing values
- 🎲 **Random Imputation**: Probability-based imputation preserving distributions
- 📊 **Manual Encoding**: Custom categorical to numerical transformation
- 🧹 **Data Validation**: Comprehensive data quality checks

### Analysis & Visualization
- 📈 **Distribution Analysis**: Visual exploration of feature distributions
- 🎨 **Matplotlib Integration**: Professional data visualizations
- 📊 **Missing Value Reports**: Detailed data quality assessments

## 📊 Dataset

**Financial Risk Assessment Dataset**
- **Total Samples**: 15,000 records
- **Features**: 11 attributes (8 numerical, 3 categorical)
- **Target**: Binary classification (Default Risk: Yes/No)
- **Missing Values**: 15% (2,250 entries per affected column)

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| Income | Numerical | Annual income of the applicant |
| Credit Score | Numerical | Credit rating (300-850) |
| Loan Amount | Numerical | Requested loan amount |
| Assets Value | Numerical | Total assets owned |
| Number of Dependents | Numerical | Family size |
| Previous Defaults | Numerical | History of defaults |
| Gender | Categorical | Male/Female/Non-binary |
| Education Level | Categorical | High School/Bachelor's/Master's/PhD |
| Marital Status | Categorical | Single/Married/Divorced |
| Employment Status | Categorical | Employed/Unemployed/Self-employed |
| **Default Risk** | **Target** | **Yes/No** |

## 🔬 Implementation Details

### ID3 Algorithm Steps

1. **Entropy Calculation**
   ```python
   H(S) = -Σ(p_i * log₂(p_i))
   ```
   - Measures uncertainty/impurity in the dataset
   - Range: 0 (pure) to log₂(n) (maximum uncertainty)

2. **Information Gain Computation**
   ```python
   IG(S, A) = H(S) - Σ((|S_v|/|S|) * H(S_v))
   ```
   - Determines best feature to split on
   - Maximizes reduction in entropy

3. **Recursive Tree Building**
   - Select feature with highest information gain
   - Create child nodes for each feature value
   - Recursively build subtrees
   - Stop when pure nodes or no features remain

### Data Preprocessing Pipeline

#### 1. Missing Value Handling

**Median Imputation** (Numerical Features)
- Applied to: Income, Credit Score, Loan Amount, Assets Value
- Method: Replace NaN with column median
- Rationale: Robust to outliers, preserves central tendency

**Random Imputation** (Categorical Features)
- Applied to: Number of Dependents, Previous Defaults
- Method: Sample from existing value distribution
- Rationale: Maintains original data distribution

#### 2. Categorical Encoding

Manual encoding mappings:
```python
Gender: {Male: 0, Female: 1, Non-binary: 2}
Education: {High School: 0, Bachelor's: 1, Master's: 2, PhD: 3}
Marital Status: {Single: 0, Married: 1, Divorced: 2}
Employment: {Unemployed: 0, Employed: 1, Self-employed: 2}
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/b2210356021/decision-tree-from-scratch.git
cd decision-tree-from-scratch

# Install required packages
pip install pandas==2.2.3 numpy==2.0.2 matplotlib==3.9.4

# Or use requirements.txt
pip install -r requirements.txt
```

## 💻 Usage

### Running the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook decision_tree_from_scratch.ipynb
```

### Quick Start Example

```python
import pandas as pd
from decision_tree import DecisionTreeID3

# Load preprocessed data
df = pd.read_csv("financial_risk_assessment.csv")

# Initialize and train
tree = DecisionTreeID3(max_depth=5)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Visualize tree
tree.print_tree()
```

## 📈 Results

### Model Performance

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| Accuracy | 85.2% | 82.7% |
| Precision | 84.1% | 81.3% |
| Recall | 86.5% | 84.2% |
| F1-Score | 85.3% | 82.7% |

### Tree Statistics
- **Total Nodes**: 47
- **Leaf Nodes**: 24
- **Maximum Depth**: 5
- **Most Important Feature**: Credit Score (IG: 0.342)

## 🎓 Technical Highlights

### Algorithm Complexity
- **Time Complexity**: O(n × m × log n) where n = samples, m = features
- **Space Complexity**: O(n × d) where d = tree depth

### Key Learnings
1. **Information Theory Application**: Practical use of entropy in ML
2. **Recursive Algorithms**: Tree traversal and construction
3. **Statistical Imputation**: Advanced missing value strategies
4. **Custom Implementation**: Deep dive into algorithm internals
5. **Data Quality**: Importance of preprocessing in ML pipelines

### Optimization Techniques
- Early stopping with max_depth parameter
- Efficient entropy calculations with vectorization
- Memory-efficient tree representation
- Pruning strategies for overfitting prevention

## 📁 Project Structure

```
decision-tree-from-scratch-financial-risk-assesment/
│
├── decision_tree_from_scratch_financial_risk_assesment.ipynb  # Main Jupyter notebook
├── financial_risk_assessment.csv     # Dataset
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT License
```

**Note:** This is a self-contained academic project. All implementation, preprocessing, and analysis are included in the main notebook for clarity and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contact

**Mehmet Oğuz Kocadere**
- 📧 Email: canmehmetoguz@gmail.com
- 💼 LinkedIn: [mehmet-oguz-kocadere](https://linkedin.com/in/mehmet-oguz-kocadere)
- 🐙 GitHub: [@memo-13-byte](https://github.com/memo-13-byte)

## 🙏 Acknowledgments

- Hacettepe University - Computer Engineering Department
- BBM 409: Machine Learning Laboratory Course
- Course Instructors for valuable guidance

---

**⭐ If you found this project helpful, please consider giving it a star!**

**🔗 Related Projects**
- [Naive Bayes Sentiment Analysis - Amazon Reviews Analysis](https://github.com/memo-13-byte/Naive-Bayes-Sentiment-Analysis-Amazon-Reviews-Analysis)
- [Bird Species Classifier CNN](https://github.com/memo-13-byte/bird-species-classifier-cnn)
- [RepoWise - A RAG based Repository Chat Bot](https://github.com/memo-13-byte/A-RAG-based-Repository-Chat-Bot)