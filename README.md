# Building-ML-Foundations-for-Smart-Air-Purifier-Project

This project is a **practice exercise** where I apply **scikit-learn** to a synthetic fraud detection dataset.  
The goal is not to build a production fraud detection system, but to practice the **end-to-end machine learning workflow in Python**.

I plan to transfer these skills into my main **Air Purifier ML Project**, where I will use machine learning to predict when an air purifier should activate based on air quality and energy efficiency.

---

## 📌 Project Goals
- Load and preprocess data using **pandas**.  
- Encode categorical variables with **LabelEncoder**.  
- Train a **Logistic Regression** model for binary classification.  
- Explore basic **correlations** between features and the fraud label.  
- Create an **interactive script** where a user enters transaction details and gets a fraud probability prediction.
  
---

## ⚙️ How It Works
1. **Load Data**  
   - CSV file is read with pandas.  
   - Categorical columns (`Location`, `TransactionType`, `JobArea`, `ItemCategory`) are converted to integers using `LabelEncoder`.  

2. **Check Correlations**  
   - `pearsonr` computes linear correlations between each feature and the fraud label.  

3. **Train Model**  
   - Logistic Regression is trained on an 80/20 split of the data.  
   - `stratify=y` ensures the class balance (fraud vs. legitimate) is preserved.  

4. **User Prediction**  
   - The script prompts the user for transaction details.  
   - Inputs are encoded with the same encoders used in training.  
   - A fraud prediction and probability are displayed.  

---

## ▶️ Example Run

--- Pearson Correlation Coefficients ---

Time: 0.028

Amount: -0.020

Age: 0.006

Location: -0.005

TransactionType: 0.053

JobArea: -0.005

ItemCategory: 0.003



=== Credit Card Fraud Checker ===

Enter transaction amount ($): 100

Enter time (seconds since first transaction): 1000

Enter user age (18-80): 25



Available options for Location: ['CA', 'FL', 'GA', 'IL', 'MI', 'NC', 'NY', 'OH', 'PA', 'TX']

Enter Location: NY



Available options for TransactionType: ['In-Store', 'Online']

Enter TransactionType: Online



Available options for JobArea: ['Education', 'Engineering', 'Finance', 'Healthcare', 'Hospitality', 'Logistics', 'Retail', 'Tech Support']

Enter JobArea: Education



Available options for ItemCategory: ['Clothing', 'Electronics', 'Furniture', 'Gas', 'Groceries', 'Pharmacy', 'Restaurants', 'Toys']

Enter ItemCategory: Clothing



--- Result ---

Legitimate transaction. (Fraud probability: 0.03)

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ml-fraud-detection-demo.git
cd ml-fraud-detection-demo
pip install -r requirements.txt
```
