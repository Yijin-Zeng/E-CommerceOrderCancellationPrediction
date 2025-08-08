# E-Commerce Order Cancellation Prediction

This is a machine learning/data science project which predicts order cancellations in e-commerce operations.
Order cancellations pose significant challenges in e-commerce, including direct revenue impact
and potential unnecessary operational costs. It is therefore important to understand cancellation patterns,
and make accurate cancellation predictions.

One challenge of this project is to deal with highly imbalanced dataset. Most orders
are not cancelled and there are only a small subset of cancelled instances.

## Dataset Overview

The dataset contains **4,000 orders** with **18 attributes** including:
- **Order Details**: Order ID, total price, lead time, delivery date
- **Customer Information**: Customer segment, platform used
- **Payment Data**: Payment type (credit/debit)
- **Logistics**: Store assignment, delivery scheduling
- **Target Variable**: Order status (Canceled/Not Canceled)


### Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Statistical summaries and data quality assessment
   - Correlation analysis and feature relationships
   - Data visualization and pattern identification
   - Handling data anomalies and missing values

2. **Data Preprocessing**
   - Categorical variable encoding
   - Feature scaling and normalization
   - Train-test splitting with stratification
   - Addressing class imbalance

3. **Model Development**
   - **Baseline Model**: Random prediction for performance comparison
   - **Logistic Regression**: Linear classification with class weighting
   - **Random Forest**: Ensemble method with hyperparameter tuning

4. **Model Evaluation**
   - Cross-validation for robust performance estimation
   - Multiple metrics: Accuracy, Precision, Recall, F1-Score
   - ROC curves and precision-recall analysis
   - Threshold optimization for business requirements

## Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline (Random) | 93.2% | 0.04 | 0.50 | 0.07 |
| Logistic Regression | 94.8% | 0.35 | 0.27 | 0.30 |
| Random Forest | 95.1% | 0.42 | 0.31 | 0.36 |

### Real-word Impact 

(images/compareWithBaseLine.png)


Among the three models considered, the random forest model performs the best (See Model Performance Summary). Its precision score is almost three times higher than that of the baseline model. While this number may not seem impressive at first glance, it demonstrates the model's practical utility. For example, suppose there are 1,000 orders to be predicted. Based on the dataset, it is reasonable to assume that around 4% of orders will be canceled. If both the baseline and random forest models are used for prediction with a recall score of 80%, they will each successfully predict about 32 orders that are likely to be canceled. Given the baseline model's precision score of 5.2%, it will make (1-0.052)/0.052 * 32 = 583 false positive prediction. In contrast, the random forest model, with its precision score of 14.8%, will make (1-0.148)/0.148 * 32 = 184 false positive predictions. Therefore, by using the random forest model instead of the baseline model, we could reduce the number of false positive predictions by 399 per 1,000 predictions.