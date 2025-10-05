# 🧠 LSTM Basic & Text Classification Projects

## 📋 Project Overview

This repository contains two comprehensive machine learning implementations:

1. 🧠 **LSTM from Scratch** — Time Series Forecasting using a custom-built LSTM implementation.  
2. 📝 **Text Classification with LSTM** — Genre prediction using deep learning.

Both projects demonstrate core AI concepts with **end-to-end implementations**, including data preparation, model design, training, and evaluation.

---

## 🚀 Project 1: LSTM from Scratch — Time Series Forecasting

### 🌟 Features
- Custom LSTM implementation using **pure NumPy**
- Time series forecasting on **synthetic data**
- Complete training pipeline with **forward & backward propagation**
- **Gradient clipping** and **early stopping**
- Visualization of training progress and predictions

### 🧩 Architecture

| Parameter | Description |
|------------|-------------|
| Input Size | 1 feature |
| Hidden Units | 50 LSTM cells |
| Output | 1 (regression) |
| Lookback Window | 10 time steps |

### 📈 Results

| Metric | Value |
|---------|--------|
| Test MSE | ~0.001–0.01 |
| Training Epochs | 100–200 |
| Outcome | Effective pattern learning in time series |

---

## 🚀 Project 2: Text Classification with LSTM

### 🌟 Features
- Multi-class text classification (**Education**, **Finance**, **Politics**, **Sports**)
- **Embedding Layer** for sequence processing
- Multiple feature extraction methods (**BoW**, **TF-IDF**, **Word Embeddings**)
- Comparison with **traditional ML models**
- Comprehensive **evaluation metrics & visualizations**

### 🧩 Architecture


| Parameter | Description |
|------------|-------------|
| Vocabulary Size | 5,000 words |
| Embedding Dimension | 100 |
| LSTM Units | 64 |
| Output | 4 classes (softmax activation) |

### 📈 Results
| Metric | Value |
|---------|--------|
| Accuracy | 85–95% |
| Comparison | LSTM outperforms traditional methods |
| Observation | Effective text pattern recognition |

---

## 🛠️ Technical Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| **Language** | Python 3.7+ |
| **Core Libraries** | NumPy, Pandas, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Deep Learning** | TensorFlow / Keras |
| **NLP** | NLTK |

### ⚙️ Key Algorithms
- LSTM Networks (custom + Keras)
- Random Forest
- Gradient Boosting
- Feature extraction: BoW, TF-IDF, Embeddings

---

## 📊 Dataset Information

### 📈 LSTM Project
| Attribute | Details |
|------------|----------|
| Type | Synthetic time series data |
| Samples | 1,000 generated points |
| Features | Sine wave with noise |
| Task | Time series forecasting |

### 📝 Text Classification Project
| Attribute | Details |
|------------|----------|
| Type | Synthetic text data |
| Categories | Education, Finance, Politics, Sports |
| Samples | 160+ text samples |
| Task | Multi-class classification |

---

## 🎯 Key Learning Outcomes

By exploring these projects, you’ll gain a deep understanding of:

### 🧠 Deep Learning Concepts
- LSTM architecture and mathematics  
- Forward & backward propagation  
- Gradient computation and weight updates  
- Embedding layers for text data  

### 🤖 Machine Learning Pipeline
- Data preprocessing and normalization  
- Feature engineering (BoW, TF-IDF, embeddings)  
- Model training, validation, and tuning  
- Performance visualization  

### 🧰 Practical Skills
- Implementing neural networks **from scratch**  
- Handling sequential data (time series + text)  
- Hyperparameter tuning and early stopping  
- Comparing deep learning vs. classical ML models  

---

## 📁 Project Structure

```bash
lstm-projects/
│
├── LSTM_from_scratch/
│   ├── lstm_scratch.ipynb          # Custom LSTM implementation
│   ├── synthetic_data.csv          # Generated dataset
│   ├── results/                    # Visualization outputs
│   └── plots/                      # Training & prediction graphs
│
├── Text_Classification_LSTM/
│   ├── text_classification.ipynb   # Text classification with Keras LSTM
│   ├── text_data.csv               # Synthetic dataset
│   ├── metrics.png                 # Evaluation metrics visualization
│   └── confusion_matrix.png        # Confusion matrix
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 👨‍💻 Author
### Syed Danial Asghar Zaidi <br>
🎓 M.S. in Artificial Intelligence, Beijing Institute of Technology <br>
💼 Focus areas: Deep Learning, NLP, Time Series Modeling <br>
📧 aghadani3@gmail.com


---
