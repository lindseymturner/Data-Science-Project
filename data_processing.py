import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# From our logistic regression implementation:
from logistic_regression import train_logistic_regression, acc, confusion_matrix, pred, prob_pos
from sklearn.metrics import roc_curve, auc
#import kagglehub

df = pd.read_csv('data.csv')

# dimensions: 
# print(f'DataFrame shape: {df.shape}') # DataFrame shape: (6819, 96)

# identify missing vals: 
missing_vals = df.isnull().sum()
# print('Missing values per column:')
# print(missing_vals[missing_vals > 0])
#### There are no missing values in the dataset. This is good. 

# Check classification class (Bankrupcy or Not Bankrupcy) : 
class_counts = df['Bankrupt?'].value_counts()
# print('Class distribution:')
# print(class_counts)
# Class distribution: 
# 0    6525  --- Not Bankrupt
# 1     294  --- Bankrupt

# Seperate features and labels:
X = df.drop(columns=['Bankrupt?']) # All but bankrupt column
y = df['Bankrupt?']
# ------------------------------------------------------------------------------------------ #
## Remove constant features (std = 0):
X = X.loc[:, X.std() != 0]


# We remove 1 constant feature in this dataset. Since it makes it so it is high on information gain but on the ROC curve is is just a diagonal loine. THis is inaccurate. 
# This fixes out issues. remvoed feature 

## Normalize features:
X = (X - X.mean()) / X.std()
# ------------------------------------------------------------------------------------------ #
# Train-test split (80/20 split): Used - sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train and test data:
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# print('Train data shape:', train_data.shape) # Train data shape: (5455, 96))
# print('Test data shape:', test_data.shape) # Test data shape: (1364, 96))

# ------------------------------------------------------------------------------------------ #
## Now we're moving onto considering informaition gain/ feature importance. 

# Calcualte entropy: 
def entropy(y):
    """Calulate the entropy of a label y"""
    # Use pandas to get counts and noamrlize the data (makes this process simple): 
    val_counts = y.value_counts(normalize=True)
    return -sum(val_counts * np.log2(val_counts + 1e-9))

def information_gain(feature, target):
    """Since we have a continuous feature data we need to make it descrete and then calucalte the info gain."""
    # Discretize if continuous 
    # Basically what this does is it splits the continuous feature into 3 bins (low, medium, high) therefore it is discrete. 
    #print(feature)
    #print("--------------")
    # Pandas way to cantinerize features into 3 (q=3) bins:
    if pd.api.types.is_numeric_dtype(feature):
        feature = pd.qcut(feature, q=3, labels=False, duplicates='drop')
    # For each features this will print out the feature values it has (discrete values now) they correspond to the influence on the classification.
    #print(feature) # Uncomment to see feature values in bins (0, 1, 2).
    ent = entropy(target)
    ent_after = 0
    for val in feature.unique():
        sub = target[feature == val]
        ent_after += (len(sub) / len(target)) * entropy(sub)
    return ent - ent_after

# Calculate information gain for each feature
info_gains = {}
for col in X.columns:
    info_gains[col] = information_gain(X[col], y)
# Sort features by information gain
sorted_ig = sorted(info_gains.items(), key=lambda x: x[1], reverse=True)

print(f"Top features by Information Gain:{sorted_ig[:10]}")

# Features with information gain > 0.03
# c = 0 
# ig_score = 0
# for x,y in sorted_ig:
#     if .01 > y > ig_score:
#         c += 1
# print(f'Feature with info gain > {ig_score} : {c}')

# Findings : 
# Top 10 features by Information Gain:[(' Net Income to Total Assets', 0.03864504845482147), (' Persistent EPS in the Last Four Seasons', 0.0370461741012342), (' Retained Earnings to Total Assets', 0.036285097758239765), (' Net profit before tax/Paid-in capital', 0.03528845361581437), (' Per Share Net profit before tax (Yuan ¥)', 0.03458973655368469), (' Total income/Total expense', 0.034411402819831904), (' Continuous interest rate (after tax)', 0.0342515274355307), (' ROA(B) before interest and depreciation after tax', 0.03389258382884022), (' ROA(C) before interest and depreciation before interest', 0.03307695318424453), (' Pre-tax net Interest Rate', 0.032319386028786395)]
# ------------------------------------------------------------------------------------------ #
## Now we will implement Logistic Regression using gradiesnt descent on our 80/20 split data. (OUtput 1-0). We imported this from other file : 'logistic_regression.py'.

'''
Note: We have a very imbalanced dataset (294 pos to 6525 neg). Therefore we will use a threshold of 0.3 or something of this
size in order to classify a point as positive (1). This should help us catch more of the positive cases instead of classifying everything as negative (0).
'''

# Convert to numpy arr
X_train_np = X_train.values
X_test_np = X_test.values
# Add column of ones : 
X_train_np = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
X_test_np = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])

y_train_np = y_train.values
y_test_np = y_test.values

# Train logistic regression
w = train_logistic_regression(X_train_np, y_train_np, alpha=0.01, tolerance=1e-6, max_iter=1000)
# Predict (Vary threshold for differnt findings) :
y_pred = pred(X_test_np,w,threshold=0.5)
# Accuracy 
accuracy, correct, total = acc(y_test_np, y_pred)
print("Accuracy:", accuracy)
# Confusion matrix from logistic regression
tp, tn, fp, fn = confusion_matrix(y_test_np, y_pred)
print("TP, TN, FP, FN:", tp, tn, fp, fn)

# Confusion Matrix findings: 
# When threshold = 0.5 : 
# Accuracy: 0.9648093841642229
# TP, TN, FP, FN: 9 1307 13 35

# See the probabilites. If max prob is lower than threshold then we know why we have 0 TPs. This is because all are getting classified as negative.
probs = [prob_pos(x, w) for x in X_test_np]
print(max(probs))
print(min(probs))

# ------------------------------------------------------------------------------------------ #
## Plotting ROC curve for top featuers by information gain.
# Using sklearn metrics for ROC curve and AUC calculation.
# Change if wanted: We will plot ROC curves for the top hwoever many features by information gain.
#                                                   Change here for more features
top_features_list = [(name, score) for name, score in sorted_ig[:10] if name in X.columns]
top_features_list = top_features_list[:10]  # Take top x non-constant features

plt.figure(figsize=(10, 8)) # Set fig size
print('===============')
c = 0 
for feature_name, ig_score in top_features_list:
    # Get the index of the feature in the original dataframe
    feat_idx = X.columns.get_loc(feature_name)
    
    # Extract just that feature for Train and Test
    # Note: We need column 0 for the bias. 
    X_train_feat = X_train_np[:, [0, feat_idx + 1]]
    X_test_feat = X_test_np[:, [0, feat_idx + 1]]
    
    # Train a logistic regression model specifically for this feature
    w_feat = train_logistic_regression(X_train_feat, y_train_np, alpha=0.01, tolerance=1e-3, max_iter=1000)
    
    # Get probs
    probs_feat = [prob_pos(x, w_feat) for x in X_test_feat]
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_test_np, probs_feat)
    feat_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=1.5, label=f'{feature_name} (AUC = {feat_auc:.3f})')
    print(f'Feature: {feature_name}, Information Gain: {ig_score:.6f}, AUC: {feat_auc:.6f}')

print(c)

print('===============')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Top Features by Information Gain with AUC')
# This command generates the box showing your labels
plt.legend(loc="lower right", fontsize='small') 
plt.show()

# Findings:

'''
DATA ON TOP 90 FEATURES BY INFORMATION GAIN AND AUC:

Feature:  Net Income to Total Assets, Information Gain: 0.038645, AUC: 0.898915
Feature:  Persistent EPS in the Last Four Seasons, Information Gain: 0.037046, AUC: 0.901920
Feature:  Retained Earnings to Total Assets, Information Gain: 0.036285, AUC: 0.888361
Feature:  Net profit before tax/Paid-in capital, Information Gain: 0.035288, AUC: 0.901007
Feature:  Per Share Net profit before tax (Yuan ¥), Information Gain: 0.034590, AUC: 0.898666
Feature:  Total income/Total expense, Information Gain: 0.034411, AUC: 0.881921
Feature:  Continuous interest rate (after tax), Information Gain: 0.034252, AUC: 0.129356
Feature:  ROA(B) before interest and depreciation after tax, Information Gain: 0.033893, AUC: 0.881913
Feature:  ROA(C) before interest and depreciation before interest, Information Gain: 0.033077, AUC: 0.882283
Feature:  Pre-tax net Interest Rate, Information Gain: 0.032319, AUC: 0.128289
Feature:  After-tax net Interest Rate, Information Gain: 0.032234, AUC: 0.129855
Feature:  ROA(A) before interest and % after tax, Information Gain: 0.032206, AUC: 0.883919
Feature:  Net Income to Stockholder's Equity, Information Gain: 0.030201, AUC: 0.833523
Feature:  Total debt/Total net worth, Information Gain: 0.029971, AUC: 0.833437
Feature:  Borrowing dependency, Information Gain: 0.029333, AUC: 0.805294
Feature:  Debt ratio %, Information Gain: 0.029255, AUC: 0.835727
Feature:  Equity to Liability, Information Gain: 0.029239, AUC: 0.835761
Feature:  Net worth/Assets, Information Gain: 0.029223, AUC: 0.835727
Feature:  Quick Ratio, Information Gain: 0.028807, AUC: 0.834358
Feature:  Non-industry income and expenditure/revenue, Information Gain: 0.026993, AUC: 0.876463
Feature:  Degree of Financial Leverage (DFL), Information Gain: 0.026307, AUC: 0.240461
Feature:  Liability to Equity, Information Gain: 0.025413, AUC: 0.767579
Feature:  Interest Expense Ratio, Information Gain: 0.025250, AUC: 0.767002
Feature:  Net Value Growth Rate, Information Gain: 0.025002, AUC: 0.148640
Feature:  Quick Assets/Current Liability, Information Gain: 0.024418, AUC: 0.818363
Feature:  Net Value Per Share (A), Information Gain: 0.023905, AUC: 0.833936
Feature:  Net Value Per Share (B), Information Gain: 0.023886, AUC: 0.834625
Feature:  Interest Coverage Ratio (Interest expense to EBIT), Information Gain: 0.023567, AUC: 0.752720
Feature:  Current Ratio, Information Gain: 0.023353, AUC: 0.184401
Feature:  Current Liability to Current Assets, Information Gain: 0.023353, AUC: 0.814859
Feature:  Operating profit/Paid-in capital, Information Gain: 0.023199, AUC: 0.835468
Feature:  Operating Profit Per Share (Yuan ¥), Information Gain: 0.023160, AUC: 0.835787
Feature:  Net Value Per Share (C), Information Gain: 0.023005, AUC: 0.834143
Feature:  Current Liabilities/Equity, Information Gain: 0.022250, AUC: 0.742476
Feature:  Current Liability to Equity, Information Gain: 0.022250, AUC: 0.742476
Feature:  Operating Profit Rate, Information Gain: 0.021828, AUC: 0.189954
Feature:  Operating profit per person, Information Gain: 0.017972, AUC: 0.794249
Feature:  Working Capital to Total Assets, Information Gain: 0.017535, AUC: 0.795041
Feature:  Current Liability to Assets, Information Gain: 0.017269, AUC: 0.771505
Feature:  Cash flow rate, Information Gain: 0.016596, AUC: 0.762087
Feature:  Operating Funds to Liability, Information Gain: 0.016271, AUC: 0.760675
Feature:  Cash/Current Liability, Information Gain: 0.014669, AUC: 0.215401
Feature:  Realized Sales Gross Margin, Information Gain: 0.014611, AUC: 0.767906
Feature:  Gross Profit to Sales, Information Gain: 0.014512, AUC: 0.771901
Feature:  Operating Gross Margin, Information Gain: 0.014478, AUC: 0.771918
Feature:  Cash/Total Assets, Information Gain: 0.013802, AUC: 0.795368
Feature:  Working capitcal Turnover Rate, Information Gain: 0.013348, AUC: 0.707404
Feature:  CFO to Assets, Information Gain: 0.012686, AUC: 0.743991
Feature:  Cash Flow Per Share, Information Gain: 0.012596, AUC: 0.754218
Feature:  No-credit Interval, Information Gain: 0.011580, AUC: 0.688378
Feature:  Interest-bearing debt interest rate, Information Gain: 0.010922, AUC: 0.293638
Feature:  Inventory/Working Capital, Information Gain: 0.009853, AUC: 0.608213
Feature:  Working Capital/Equity, Information Gain: 0.009760, AUC: 0.675207
Feature:  Long-term fund suitability ratio (A), Information Gain: 0.009238, AUC: 0.719155
Feature:  Cash Flow to Liability, Information Gain: 0.008997, AUC: 0.636295
Feature:  Tax rate (A), Information Gain: 0.008922, AUC: 0.755277
Feature:  Continuous Net Profit Growth Rate, Information Gain: 0.008061, AUC: 0.695833
Feature:  After-tax Net Profit Growth Rate, Information Gain: 0.007999, AUC: 0.294396
Feature:  Total Asset Growth Rate, Information Gain: 0.007783, AUC: 0.623140
Feature:  Regular Net Profit Growth Rate, Information Gain: 0.007770, AUC: 0.297392
Feature:  Cash Turnover Rate, Information Gain: 0.007676, AUC: 0.484650
Feature:  Equity to Long-term Liability, Information Gain: 0.007658, AUC: 0.654838
Feature:  Cash Flow to Total Assets, Information Gain: 0.007311, AUC: 0.657817
Feature:  Realized Sales Gross Profit Growth Rate, Information Gain: 0.006855, AUC: 0.279141
Feature:  Total Asset Return Growth Rate Ratio, Information Gain: 0.006452, AUC: 0.672185
Feature:  Cash Flow to Sales, Information Gain: 0.006220, AUC: 0.686536
Feature:  Allocation rate per person, Information Gain: 0.005783, AUC: 0.369749
Feature:  Cash Reinvestment %, Information Gain: 0.005557, AUC: 0.655088
Feature:  Total Asset Turnover, Information Gain: 0.005163, AUC: 0.668173
Feature:  Cash Flow to Equity, Information Gain: 0.004836, AUC: 0.631939
Feature:  Operating Profit Growth Rate, Information Gain: 0.004679, AUC: 0.327626
Feature:  Inventory and accounts receivable/Net value, Information Gain: 0.004640, AUC: 0.624397
Feature:  Quick Assets/Total Assets, Information Gain: 0.003951, AUC: 0.667166
Feature:  Total expense/Assets, Information Gain: 0.003773, AUC: 0.626618
Feature:  Long-term Liability to Current Assets, Information Gain: 0.003648, AUC: 0.360976
Feature:  Revenue Per Share (Yuan ¥), Information Gain: 0.003606, AUC: 0.638809
Feature:  Total assets to GNP price, Information Gain: 0.003154, AUC: 0.594766
Feature:  Inventory/Current Liability, Information Gain: 0.002749, AUC: 0.405389
Feature:  Fixed Assets Turnover Frequency, Information Gain: 0.002722, AUC: 0.583480
Feature:  Fixed Assets to Assets, Information Gain: 0.002546, AUC: 0.579804
Feature:  Current Asset Turnover Rate, Information Gain: 0.002288, AUC: 0.575465
Feature:  Current Assets/Total Assets, Information Gain: 0.001244, AUC: 0.583247
Feature:  Current Liabilities/Liability, Information Gain: 0.001201, AUC: 0.539997
Feature:  Current Liability to Liability, Information Gain: 0.001201, AUC: 0.539997
Feature:  Accounts Receivable Turnover, Information Gain: 0.001120, AUC: 0.600430
Feature:  Average Collection Days, Information Gain: 0.000981, AUC: 0.382197
Feature:  Net Worth Turnover Rate (times), Information Gain: 0.000963, AUC: 0.495541
Feature:  Quick Asset Turnover Rate, Information Gain: 0.000821, AUC: 0.603883
Feature:  Research and development expense rate, Information Gain: 0.000625, AUC: 0.581878
Feature:  Contingent liabilities/Net worth, Information Gain: 0.000541, AUC: 0.551352
'''

# ------------------------------------------------------------------------------------------ #
# Precision, Recall Calculation
precision = tp / (tp + fp) 
recall = tp / (tp + fn) 

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
