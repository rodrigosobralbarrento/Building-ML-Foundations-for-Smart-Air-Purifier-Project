import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# FEATURES controls which columns are used by the model and their order.
# the same order must be used at both training time and prediction time.
FEATURES = ['Time', 'Amount', 'Age', 'Location', 'TransactionType', 'JobArea', 'ItemCategory']

# LABEL is the target we want to predict. In this dataset 1 means fraud and 0 means legitimate.
LABEL = 'Class'

def load_data(path):
    """
    Load the credit card transactions dataset from CSV, 
    encode categorical variables with LabelEncoder, 
    and return a cleaned DataFrame ready for training along with the fitted encoders.
    
    Parameters:
        path (str) : path to the CSV file.
    
    Returns:
        df_train (pandas.DataFrame) :  DataFrame containing numeric and encoded categorical features plus the target label.
        
        le_loc, le_txn, le_job, le_cat (LabelEncoder) : Fitted LabelEncoder objects for Location, TransactionType, JobArea, and ItemCategory.
    """
    
    # read the dataset into a pandas DataFrame
    df = pd.read_csv(path)

    # create one LabelEncoder per categorical feature.
    # each encoder learns a mapping from category string to integer code.
    le_loc = LabelEncoder()
    le_txn = LabelEncoder()
    le_job = LabelEncoder()
    le_cat = LabelEncoder()

    # fit each encoder on its column and create encoded columns.    
    df['Location_enc'] = le_loc.fit_transform(df['Location'])
    df['TransactionType_enc'] = le_txn.fit_transform(df['TransactionType'])
    df['JobArea_enc'] = le_job.fit_transform(df['JobArea'])
    df['ItemCategory_enc'] = le_cat.fit_transform(df['ItemCategory'])

    # making a copy to not overwrite
    df_train = df.copy()
    
    # overwrite the original categorical columns with their encoded versions so that
    # the feature names stay familiar, but the values are numeric.
    df_train['Location'] = df_train['Location_enc']
    df_train['TransactionType'] = df_train['TransactionType_enc']
    df_train['JobArea'] = df_train['JobArea_enc']
    df_train['ItemCategory'] = df_train['ItemCategory_enc']

    # keep only the training features and the label. This ensures a clean design matrix X
    # and a clear target y for the model.
    df_train = df_train[['Time', 'Amount', 'Age', 'Location',
                         'TransactionType', 'JobArea', 'ItemCategory', 'Class']]

    # return the prepped data and the fitted encoders
    return df_train, le_loc, le_txn, le_job, le_cat

def print_correlations(df_enc):
    """
    Print Pearson correlation coefficients between each feature and the target label.
    
    Parameters:
        df_enc (pandas.DataFrame) : DataFrame with encoded features and the target label.
    """
    
    print("\n--- Pearson Correlation Coefficients ---")
    
    # pearsonr returns two values: correlation coefficient and p value.
    # ignoring p-value as it is not used
    for col in FEATURES:
        coef, _ = pearsonr(df_enc[col].astype(float), df_enc[LABEL].astype(float))
        print(f"{col}: {coef:.3f}")

def train_model(df_enc):
    """
    Train a logistic regression model to predict fraud from encoded features.
    
    Parameters:
        df_enc (pandas.DataFrame) : DataFrame with encoded features and the target label.
    
    Returns:
        model (LogisticRegression) : Trained logistic regression classifier.
    """
    
    # split features and labels
    X = df_enc[FEATURES]
    y = df_enc[LABEL]
    
    # create train and test subsets
    # using same split every run for consistency
    # preserving class balance (keeping training set and test set on same ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # create a logistic regression model, increasing max_tier to help convergence
    model = LogisticRegression(max_iter=1000)
    
    # fit the model on training data only
    model.fit(X_train, y_train)
    
    # return trained model
    return model

def predict_user_transaction(model, le_loc, le_txn, le_job, le_cat):
    """
    Collect transaction details from the user, encode them with the fitted LabelEncoders,
    and use the trained model to predict whether the transaction is fraudulent.
    
    Parameters:
        model (LogisticRegression) : Trained logistic regression classifier.
        
        le_loc, le_txn, le_job, le_cat (LabelEncoder) : Fitted encoders to transform categorical user input into numeric codes.
    """
    
    print("\n=== Credit Card Fraud Checker ===")
    # Numeric inputs
    amount = float(input("Enter transaction amount ($): ").lower().strip())
    time_val = float(input("Enter time (seconds since first transaction): ").lower().strip())
    age = int(input("Enter user age (18-80): ").lower().strip())

    # Location
    loc_options = list(le_loc.classes_)
    print(f"\nAvailable options for Location: {loc_options}")
    loc_in = input("Enter Location: ").lower().strip()
    while loc_in not in loc_options:
        print(f"Please choose one of {loc_options}")
        loc_in = input("Enter Location: ").lower().strip()
    # transform category into integer code for model
    loc_code = int(le_loc.transform([loc_in])[0])

    # TransactionType
    txn_options = list(le_txn.classes_)
    print(f"\nAvailable options for TransactionType: {txn_options}")
    txn_in = input("Enter TransactionType: ").lower().strip()
    while txn_in not in txn_options:
        print(f"Please choose one of {txn_options}")
        txn_in = input("Enter TransactionType: ").lower().strip()
    txn_code = int(le_txn.transform([txn_in])[0])

    # JobArea
    job_options = list(le_job.classes_)
    print(f"\nAvailable options for JobArea: {job_options}")
    job_in = input("Enter JobArea: ").lower().strip()
    while job_in not in job_options:
        print(f"Please choose one of {job_options}")
        job_in = input("Enter JobArea: ").lower().strip()
    job_code = int(le_job.transform([job_in])[0])

    # ItemCategory
    cat_options = list(le_cat.classes_)
    print(f"\nAvailable options for ItemCategory: {cat_options}")
    cat_in = input("Enter ItemCategory: ").lower().strip()
    while cat_in not in cat_options:
        print(f"Please choose one of {cat_options}")
        cat_in = input("Enter ItemCategory: ").lower().strip()
    cat_code = int(le_cat.transform([cat_in])[0])

    # Build one row with the same column names as training to avoid sklearn warnings
    x_df = pd.DataFrame([[time_val, amount, age, loc_code, txn_code, job_code, cat_code]],
                        columns=FEATURES)

    # predict class label, where 1 means fraud and 0 menas legitimate
    pred = int(model.predict(x_df)[0])
    # predict probability for fraud
    # predict_proba returns an array like [prob_class_0, prob_class_1]
    prob = float(model.predict_proba(x_df)[0][1])

    print("\n--- Result ---")
    if pred == 1:
        print(f"Fraudulent transaction detected! (Probability: {prob:.2f})")
    else:
        print(f"Legitimate transaction. (Fraud probability: {prob:.2f})")

if __name__ == '__main__':
    # path to  CSV
    data_path = 'synthetic_credit_fraud_extended.csv'
    df_enc, le_loc, le_txn, le_job, le_cat = load_data(data_path)
    print_correlations(df_enc)
    model = train_model(df_enc)
    predict_user_transaction(model, le_loc, le_txn, le_job, le_cat)
