# 📰 Fake News Detection Using Machine Learning

## 📌 Project Overview
This project applies **machine learning** to classify news articles as **real or fake**.  
It uses **Logistic Regression, Random Forest, XGBoost, SVC, and an RNN (LSTM)** to analyze text and predict fake news.

## 🛠 Technologies Used
- **Python**  
- **Scikit-learn** (for ML models)  
- **TensorFlow / Keras** (for Deep Learning RNN model)  
- **Pandas & NumPy** (for data processing)  
- **Matplotlib** (for result visualization)  

## 💂️ Files in This Project
- The dataset is available on Kaggle: [Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets).
- **`fake_news.ipynb`** → Jupyter Notebook containing all the code for training and testing the models.  

## 🛠️ How to Use
1. **Install dependencies** using:  
   ```bash
   pip install numpy pandas scikit-learn tensorflow matplotlib
   ```
2. **Open the Jupyter Notebook**:  
   ```bash
   jupyter notebook fake_news.ipynb
   ```
3. **Run the notebook cells** to train models and see the results.

## 📈 Model Performance
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|------------|--------|----------|
| Logistic Regression | 98.56%    | 0.9856     | 0.9856 | 0.9856   |
| Random Forest       | 99.49%    | 0.9949     | 0.9949 | 0.9949   |
| XGBoost            | 99.54%    | 0.9954     | 0.9954 | 0.9954   |
| SVC                | 99.19%    | 0.9919     | 0.9919 | 0.9919   |
| RNN (LSTM)         | 99.63%    | 0.99       | 0.99   | 0.99     |

## 📢 Observations
- **XGBoost performed the best** with **99.54% accuracy**.  
- **SVC took longer to train** but had high performance.  
- **RNN (LSTM) achieved the highest accuracy (99.63%)** while maintaining **strong precision, recall, and F1-score** after addressing overfitting with regularization and dropout.

## 🚀 Future Improvements
- Experiment with **more advanced deep learning architectures (e.g., Transformers, BiLSTM)**.  

## 👤 Author
- **Your Name:** Thaer Hatem
- **GitHub:** [GitHub Link](https://github.com/ThaerHatem)  

## 🐜 License
This project is open-source under the **MIT License**.

