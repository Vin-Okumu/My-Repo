# Dry Eye Disease

This project stems from curiosity about what leads to dry eye disease. Here I embark on developing a model that can link underlying parameters to identify variables that best predict the incidence of dry eye disease. I explore correlations between lifestyle factors such as daily steps, sleep time, pulse measurement, blood pressure, eating and drinking habits, stress levels, medical issues such as anxiety, hypertension, asthma, etc., and any medication used with basic ocular attributes used to predict the presence of dry eye disease and ocular health in general. The dataset used herein can be used for machine learning models, statistical analysis, and clinical decision-making to enhance early detection and personalized treatment strategies for DED. It can also be used to predict another severe sleep-related disease such as insomnia which can be directly linked with ocular surface diseases (OSD).

```python
import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
```

# Data Import
```python
#Importing data
data = pd.read_csv(r"/kaggle/input/dry-eye-disease/Dry_Eye_Dataset.csv") #reads in data

# Split 'Blood pressure' into 'Systolic BP' and 'Diastolic BP'
data[['Systolic BP', 'Diastolic BP']] = data['Blood pressure'].str.split('/', expand=True).astype(float)

# Drop the original 'Blood pressure' column
data.drop(columns=['Blood pressure'], inplace=True)

data.head() #calls glimpse of data frame
```

# Checking for Normality
With the following code we check for normality in the numerical columns of a dataset (data). Normality testing is crucial in statistical analysis to determine whether data follows a normal distribution, which impacts the choice of statistical tests and modeling approaches.

```python
#Importing required libraries
import scipy.stats as stats
from scipy.stats import shapiro, normaltest, anderson # Imports necessary statistical tests from scipy.stats

numerical_cols = data.select_dtypes(include=[np.number]).columns # Extracts columns that contain numeric data from the data frame.

results = [] # Initializes a DataFrame to store results

for col in numerical_cols: # Initiates iteration through each numerical column in the data frame.
    try:
        sample_size = len(data[col].dropna()) # Drops missing values (NaN) from each column before calculating the sample size

        if sample_size <= 5000:
            
            stat, p = shapiro(data[col].dropna()) # Prompts for the use of the Shapiro-Wilk test if the sample size is <= 5000, 
            test_name = "Shapiro-Wilk" # Shapiro-Wilk test is effective for small samples.
        else:
            
            
            stat, p = normaltest(data[col].dropna()) # For a sample size > 5000, it prompts the use of D’Agostino and Pearson’s test, 
            test_name = "D'Agostino-Pearson" # D’Agostino and Pearson’s test is better suited for large datasets.

        
        results.append([col, test_name, f"{stat:.4f}", f"{p:.4f}"]) # Stores results in a list
        
    except Exception as e:
        print(f"Error processing {col}: {e}") # Catches and prints any errors that may occur during processing.

# Converting the results list into a Pandas DataFrame with appropriate column names.
results_df = pd.DataFrame(results, 
                          columns=["Variable", "Test", 
                                   "Statistic", "p-value"]) 

# Pivot table for better display

pivot_table = results_df.pivot(index="Variable", 
                               columns="Test", 
                               values=["Statistic", "p-value"]) #Creates a pivot table where: Each row represents a 
                                                                #variable and Columns are split based on the test used.
                                                                #shows Statistic and p-value.
# Fix column formatting
pivot_table.columns = [f"{stat} - {test}" for stat, 
                       test in pivot_table.columns] #Renames columns to clearly indicate which values belong to which test.

print(pivot_table) #Displays the results in a formatted table.
```
```python
# Convert DED variable from string to numeric, 

# Transforming the values in the 'Dry Eye Disease' column 
data['Dry Eye Disease'] = data['Dry Eye Disease'].map({'Y': 1, 'N': 0}) # The map() function transforms values in the 'Dry Eye Disease' column.
                                                                        # converts 'Y' to 1 and 'N' to 0

# View distribution
print(data['Dry Eye Disease'].value_counts())
print(data['Dry Eye Disease'].value_counts(normalize=True) * 100) # Makes value_counts() to return relative frequencies instead of absolute counts.
                                                                  # Multiplying by 100 converts the frequencies to percentages.
```

# Visualizing the Distribution
This helps inspect the data visualy to ensure the right method is chosen to run the model.

```python
#Importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns

for col in numerical_cols: #The loop iterates over each column in numerical_cols, assumed to be a list of numerical column names from the dataset.
    plt.figure(figsize=(12,5)) #Creates a new figure with a size of 12x5 inches 
                               #accommodates two subplots: a histogram with a KDE plot and a Q-Q plot.

    #Histogram and KDE
    plt.subplot(1,2,1) #Creates a subplot (1 row, 2 columns, selecting the first plot).
    sns.histplot(data[col].dropna(),kde=True, bins=30) #Plots a histogram for the column col, with:
                                                       #dropna(): Ensures missing values are removed.
                                                       #kde=True: Adds a Kernel Density Estimate (KDE) curve, which smooths the distribution.
                                                       #bins=30: Sets the number of bins for the histogram.
    plt.title(f"Histogram & KDE of {col}") #Sets the title to indicate which column is being visualized.

    # Q-Q Plot
    plt.subplot(1,2,2) #Creates the second subplot in the same figure.
    stats.probplot(data[col].dropna(), dist="norm", plot=plt) #Generates a Q-Q (Quantile-Quantile) plot, used to assess if a dataset follows a normal distribution.
                                                              #dist="norm": Compares the data against a normal distribution.
                                                              #plot=plt: Uses Matplotlib to display the plot.
    plt.title(f"Q-Q Plot of {col}") #Sets the title for the Q-Q plot.
    
    plt.show()
```

# Choosing between Decision Tree Model and Random Forest

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Define independent variables (X) and dependent variable (y)
x = data.drop(columns=["Dry Eye Disease"])  
y = data["Dry Eye Disease"]

# Encoding categorical variables
x = pd.get_dummies(x, drop_first=True)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
```

```python
# Standardize numerical features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
y_dt_pred = dt_model.predict(x_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_rf_pred = rf_model.predict(x_test)
```

```python
# Compute Accuracies
dt_acc = accuracy_score(y_test, y_dt_pred)
rf_acc = accuracy_score(y_test, y_rf_pred)

print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
```

#Accuracy Comparison

Decision Tree Accuracy: 0.5687 (56.87%)

Random Forest Accuracy: 0.6997 (69.97%).

It implies that Random Forest significantly outperforms Decision Tree in accuracy as it generalizes better and reduces overfitting.

```python
# Classification Reports
print("\nDecision Tree Report:")
print(classification_report(y_test, y_dt_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test, y_rf_pred))
```
# Precision, Recall, and F1-score Analysis

For Class 0 (No DED): Decision Tree precision is 0.39, with a recall of 0.42, and F1-score of 0.40.

Random Forest has a precision of 0.69, with a recall of 0.25, and F1-score of 0.37.

Random Forest has better precision but much lower recall—it correctly identifies more "No DED" cases but misses many.

For Class 1 (DED): Decision Tree has a precision of 0.68, with a recall of 0.65, and F1-score of 0.66.

Random Forest has a precision of 0.70, with a recall of 0.94, and F1-score of 0.80.

Random Forest is much better at detecting DED cases, with higher recall (0.94) meaning fewer false negatives.

```python
# Cross-Validation Comparison
dt_cv_score = np.mean(cross_val_score(dt_model, x_train, y_train, cv=5))
rf_cv_score = np.mean(cross_val_score(rf_model, x_train, y_train, cv=5))

print(f"\nDecision Tree CV Score: {dt_cv_score:.4f}")
print(f"Random Forest CV Score: {rf_cv_score:.4f}")
```

# Cross-Validation Scores

Decision Tree CV Score: 0.5661 (similar to accuracy, meaning stable performance).

Random Forest CV Score: 0.6929 (consistent with its accuracy, meaning it generalizes well).

Random Forest maintains stability across different training splits.

```python
# Compare ROC-AUC Scores
print("\nDecision Tree ROC-AUC:", roc_auc_score(y_test, dt_model.predict_proba(x_test)[:, 1]))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(x_test)[:, 1]))
```
# ROC-AUC Score (Discrimination Ability)
Decision Tree ROC-AUC: 0.5333 (Barely better than random guessing at 0.50). Random Forest ROC-AUC: 0.5936 (Better, but still not great).

Random Forest has slightly better discriminatory power than Decision Tree but still struggles with clear separation between DED and No DED cases.

# Conclusions & Next Steps:
Random Forest performs better overall, especially in predicting DED (class 1) cases with high recall. But it struggles to correctly classify "No DED" (class 0), likely due to class imbalance.

```python
# Feature Importance Comparison
plt.figure(figsize=(10, 5))
plt.bar(range(x.shape[1]), dt_model.feature_importances_, alpha=0.6, label="Decision Tree")
plt.bar(range(x.shape[1]), rf_model.feature_importances_, alpha=0.6, label="Random Forest")
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.legend()
plt.title("Feature Importance: Decision Tree vs. Random Forest")
plt.show()
```
# Key Takeaway:

Decision Tree has more balanced recall between classes but lower accuracy overall.

Random Forest strongly favors predicting incidence (high recall = 0.94), but struggles with No incidence (low recall = 0.25).

This suggests an imbalance issue—Random Forest may be biased toward classifying cases as DED (1).

# Random Forest

After testing the attributes of Decision Tree and Random Forest, we settle on the latter. Here we test the model's ability to predict incidence of Dry Eye Disease. We now perform feature importance analysis (either classification or regression) based on Gini impurity (for classification) or variance reduction (for regression). The code below evaluates which features contribute the most to predicting Dry Eye Disease (DED).

```python
#Random Forest - Feature importance based on Gini/Permutation

#Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #RandomForestClassifier and RandomForestRegressor are used to 
                                                                           #build either a classification or regression model.

from sklearn.preprocessing import LabelEncoder #LabelEncoder helps encode categorical variables into numerical values.

#Assuming 'x' is the feature matrix, and 'y' is the target variable (DED)
x = data.drop(columns=["Dry Eye Disease"])  #x being the feature matrix, contains all independent variables to predict "Dry Eye Disease".
y = data["Dry Eye Disease"] #y is the dependent variable, which we aim to predict ("Dry Eye Disease")

#Encoding categorical variables
label_encoders = {}
for col in x.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col]) #Identifies categorical columns in x, then encodes them into numerical values using LabelEncoder().
    label_encoders[col] = le  # Stores encoders in case you need to reverse the transformation later
```

```python
# Choose model type
if y.nunique() > 2:  # If y has more than two unique values, it is treated as a regression problem (continuous target).
    model = RandomForestRegressor(n_estimators=100, random_state=42)

else:  # If y has two unique values, it is a classification problem (binary target).
    model = RandomForestClassifier(n_estimators=100, random_state=42) # Creates a Random Forest model with 100 trees (n_estimators=100) for either case.

# Train model
model.fit(x, y) #The Random Forest model is trained on X (features) and y (target variable).

# Get feature importance
feature_importances = pd.DataFrame({"Feature": x.columns, "Importance": model.feature_importances_}) #model.feature_importances_ retrieves the importance of 
                                   #each feature in predicting y, based on Gini impurity (for classification) or variance reduction (for regression).

feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Selected features
from sklearn.feature_selection import RFE

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(rf_model, n_features_to_select=15)  # Adjust the number of features as needed
rfe.fit(x, y)

selected_features = x.columns[rfe.support_]

# Display top features
print(feature_importances.head(15))  # Show top 15 most important features affecting the prediction of Dry Eye Disease.
```
# Model Training

# Model Evaluation
```python
# Predictions using Random Forest model
y_pred = rf_model.predict(x_test)
y_prob = rf_model.predict_proba(x_test)[:, 1]  # Probability for the positive class

y_pred = rf_model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

# Overall Model Performance
Accuracy = 0.69975 (~70%), meaning the model correctly classifies approximately 70% of all cases.

However, accuracy alone can be misleading if the classes are imbalanced (i.e., more instances of 1 than 0).

# Class-wise Performance
The training dataset has two classes: Class 0 (No Dry Eye Disease) with 1,393 samples, and Class 1 (Dry Eye Disease) with 2,607 samples.

In terms of precision, 69% of predicted No Dry Eye Disease cases are classified as No Dry Eye Disease, and a further 70% of predicted Dry Eye Disease cases as Dry Eye Disease.

In terms of recall, the model finds 94% of actual Dry Eye Disease cases but only 25% of actual No Dry Eye Disease cases.

With an f1 score of 0.80 for Dry Eye Disease cases, the model can be considered good as it balances precision and recall.

However, with an f1 score of only 0.37 for No Dry Eye Disease, the model is not as good because of poor recall.

```python
# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))
```
```python
# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Dry Eye Disease", "Dry Eye Disease"], 
            yticklabels=["No Dry Eye Disease", "Dry Eye Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

```python
# ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
```

```python
# Model Training
x_selected = x[selected_features]  # Use selected features
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

# Evaluate performance
print(classification_report(y_test, y_pred))
```

# Prediction of Dry Eye Disease Incidence

After testing the model, we now create a sample data frame with variables included in the selected features to check how well the model predicts a patient's likelyhood of presenting with Dry Eye Disease, given their attributes.

```python
new_patient_data = pd.DataFrame({
    'Physical activity': [190, 201, 120, 60],
    'Average screen time': [1, 10, 8, 9],
    'Sleep duration': [8, 6, 7, 4],
    'Systolic BP': [181, 170, 131, 140],
    'Height': [162, 150, 131, 160],
    'Weight': [91, 60, 51, 80],
    'Heart rate': [51, 70, 61, 80],
    'Diastolic BP': [51, 40, 41, 40],
    'Age': [32, 40, 25, 50],
    'Daily steps': [1000, 1200, 1500, 800],
    'Sleep quality': [1, 2, 2, 2],
    'Stress level': [1, 3, 4, 2],
    'Discomfort Eye-strain': [1, 0, 0, 1],
    'Itchiness/Irritation in eye': [0, 0, 1, 0],
    'Redness in eye': [1, 1, 0, 1]
})
# Ensure it has the same feature columns as in training
new_patient_data = new_patient_data[selected_features]

# Predict
ded_prediction = rf_model.predict(new_patient_data)
ded_probability = rf_model.predict_proba(new_patient_data)[:, 1]

# Add predictions to DataFrame
new_patient_data['DED_Prediction'] = ded_prediction
new_patient_data['DED_Probability'] = ded_probability

print(new_patient_data)
```




