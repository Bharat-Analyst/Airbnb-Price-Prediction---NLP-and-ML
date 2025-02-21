# **Airbnb Price Prediction & Text-Based Analysis**
**Machine Learning | NLP | BERT | XGBoost | Random Forest | Regression Models**

## **Project Overview**
This project leverages **machine learning and natural language processing (NLP)** to predict Airbnb listing prices based on structured data (amenities, location, reviews) and unstructured textual descriptions. Using **XGBoost, Random Forest, and Linear Regression**, along with **BERT embeddings**, the model enhances price prediction by incorporating semantic insights from the listing descriptions.

---

## **Table of Contents**
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Text Processing with BERT](#text-processing-with-bert)
- [Results & Insights](#results--insights)
- [Future Enhancements](#future-enhancements)

---

## **Features**
✅ Predicts Airbnb listing prices using **machine learning models** (XGBoost, Random Forest, Linear Regression)  
✅ Uses **BERT-based embeddings** for text processing to extract semantic meaning from property descriptions  
✅ **Handles missing data** and automates feature engineering for optimal model performance  
✅ **Hyperparameter tuning** with **GridSearchCV** for optimal performance  
✅ Generates **explainable insights** into pricing based on listing attributes and reviews  

---

## **Technologies Used**
- **Python** (Pandas, NumPy, Seaborn, Matplotlib, Sklearn, Transformers)  
- **Machine Learning Models** (XGBoost, Random Forest, Linear Regression)  
- **NLP & BERT** (Hugging Face Transformers, Tokenization, Text Embeddings)  
- **Data Processing** (Feature Engineering, Missing Value Handling, Label Encoding)  
- **Model Evaluation** (MAE, RMSE, R² Score, Cross-Validation)  

---

## **Installation**

Clone the repository:  
```bash
git clone https://github.com/your-repo/Airbnb-Price-Prediction.git
cd Airbnb-Price-Prediction
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## **Data Preprocessing**
- **Handling Missing Values:** Missing features such as `bathrooms`, `bedrooms`, and `reviews` are filled using median imputation.  
- **Encoding Categorical Features:** Label encoding is applied to categorical variables to prepare them for machine learning models.  
- **Feature Selection:** Non-essential fields (`id`, `thumbnail_url`, `zipcode`) are removed for cleaner input data.  

---

## **Model Training & Evaluation**
### **1. Linear Regression**
```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
```

### **2. Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
```

### **3. XGBoost Regressor**
```python
from xgboost import XGBRegressor

xgb = XGBRegressor(objective="reg:squarederror")
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
```

📌 **Evaluation Metrics Used:**  
- **Mean Absolute Error (MAE)**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score (Coefficient of Determination)**  

---

## **Text Processing with BERT**
To enhance predictions, **BERT embeddings** were extracted from the Airbnb listing descriptions:  

```python
from transformers import AutoModel, BertTokenizerFast

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

output = tokenizer(example_text, return_tensors="pt")
embedding = model(**output).last_hidden_state
```

🔹 **BERT embeddings were merged with structured features** to create an enriched dataset for improved price prediction accuracy.  

---

## **Results & Insights**
- **XGBoost outperformed other models**, achieving the lowest **MAE and RMSE scores**.  
- **Adding textual embeddings from BERT** improved **model performance** by better understanding listing quality.  
- **Feature importance analysis showed** that **location, amenities, and review scores** heavily influenced pricing.  

---

## **Future Enhancements**
🚀 **Expand AI capabilities:** Integrate **Generative AI (LLMs)** for **automated listing descriptions** and price suggestions.  
🚀 **Deploy as an API:** Convert this model into an **interactive web app using Flask or FastAPI**.  
🚀 **Integrate real-time pricing data:** Fetch live Airbnb data to continuously update and refine predictions.  
