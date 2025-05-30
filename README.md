# Cost-Sensitive Classification of Online Orders – EECS 4412 Assignment 3 (Q5)

This project addresses a real-world classification problem involving **30,000 online purchase orders**, each labeled as either **"high-risk"** or **"low-risk"**. The task was to build a model capable of classifying **20,000 new orders**, with the goal of minimizing the overall **misclassification cost**, based on the following cost matrix:

| Actual \ Predicted | High Risk | Low Risk |
|--------------------|-----------|----------|
| High Risk          | 0         | 50       |
| Low Risk           | 5         | 0        |

---

## 🔧 1. Handling Missing Values

### 🧮 Numerical Features
- Used `SimpleImputer` with **mean** strategy to fill in missing values.
- This preserves the overall distribution of the feature without introducing skew.

### 🗂️ Categorical Features
- Used `SimpleImputer` with **most_frequent** strategy.
- Replacing missing values with the most common category ensures realistic representation of attributes like `"customer_region"`.

> Real-world datasets often have missing values due to user error, incomplete submissions, or collection issues. Proper imputation is necessary to avoid inaccuracies or crashes during model training.

---

## 🤖 2. Classification Algorithm – XGBoost

We used **XGBoost** for its effectiveness in handling:
- High-dimensional data
- Class imbalance
- Custom class weighting

### ✅ Why XGBoost?
1. Efficient and powerful for large-scale classification.
2. Handles imbalanced datasets well.
3. Allows custom cost-sensitive learning via `scale_pos_weight`.

---

## ⚖️ 3. Handling Class Imbalance & Misclassification Cost

### Strategy:
- Applied **cost-sensitive learning** using class weights.
- Tuned `scale_pos_weight` to heavily penalize misclassifying high-risk orders.
- Incorporated **threshold tuning** to adjust the decision boundary beyond default binary outputs.

---

## 🛠️ 4. Hyperparameter and Threshold Tuning

### ⚙️ Grid Search:
- Used `GridSearchCV` to optimize hyperparameters.
- Scored with **F1-score** (with "yes" as the positive class) due to its balanced penalization of false positives and false negatives.

### 🎯 Threshold Tuning:
- Predicted probabilities were calibrated, and a range of thresholds tested.
- Chose the threshold that **minimized total misclassification cost** on the validation set.

> Fine-tuning the threshold allowed the model to shift focus from accuracy to **business cost reduction**, ensuring high-risk customers were less likely to be misclassified.

---

## 📈 5. Model Effectiveness

- Final model: **XGBoost with optimized hyperparameters and custom threshold**
- Output predictions on `prediction.txt` followed the required format.
- The model **significantly reduced overall misclassification cost**, especially by correctly classifying high-risk cases.

---

## ✅ 6. Conclusion

We developed a robust classification solution by:
- Carefully preprocessing the data
- Choosing **XGBoost** for its strength in imbalanced, structured data
- Applying **cost-sensitive learning**
- Tuning the decision **threshold** to minimize the business impact

This approach resulted in a reliable prediction system for classifying high-risk vs. low-risk orders and demonstrated the **real-world value of precision-focused modeling** in cost-sensitive scenarios.
