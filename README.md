# Credit Card Fraud Detection Using Machine Learning and Deep Learning

## Project Overview

Credit card fraud is a significant challenge in financial services. Fraudulent transactions represent a small proportion of all transactions, making them difficult to identify without incorrectly flagging legitimate customers.

This project develops and evaluates machine learning and deep learning models to classify credit card transactions as **legitimate (Class 0)** or **fraudulent (Class 1)**. It compares traditional machine learning approaches with a neural network and investigates how class imbalance and classification thresholds affect fraud detection performance.

The primary objective is to identify fraudulent transactions while maintaining high precision, reducing false alarms without overlooking too many fraudulent transactions.

## Project Links

- **GitHub Repository:** [Insert GitHub repository URL]
- **Google Drive Deliverables:** [Insert shareable Google Drive folder URL]
- **Presentation Video:** [Insert video URL]

## Dataset

The project uses the Credit Card Fraud Detection dataset published on Kaggle by the Machine Learning Group of Université Libre de Bruxelles (ULB).

**Dataset source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original dataset contains **284,807 transactions** and 31 columns, including 30 input features and one target variable, `Class`.

| Property | Description |
|---|---|
| Total original transactions | 284,807 |
| Input features | 30 |
| Target variable | `Class` |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud prevalence | Approximately 0.173% |
| Missing values | None identified |

Most input features (`V1`–`V28`) are anonymized principal component analysis (PCA) features. The remaining inputs are `Time` and `Amount`.

The target variable is binary:

- `0`: Legitimate transaction
- `1`: Fraudulent transaction

### Data Preparation

The following preprocessing steps were performed:

1. Inspected the dataset structure, class distribution, feature distributions, and correlations.
2. Investigated duplicate records and removed 1,081 exact duplicate rows from the modelling dataset.
3. Preserved the original dataset separately from the deduplicated modelling data.
4. Used stratified splitting to create training, validation, and test sets while preserving the class distribution.
5. Standardized features for Logistic Regression and the neural network, fitting the scalers only on the training data.
6. Used the training data to fit the models and the validation data to compare models and select classification thresholds.

After duplicate removal, the modelling dataset contained **283,726 transactions**, including 473 fraudulent transactions.

The data was split into training, validation, and test sets, with approximately 70%, 15%, and 15% of the data allocated to each set, respectively.

## Methodology

### 1. Exploratory Data Analysis

Exploratory analysis examined class imbalance, transaction amounts, duplicate records, and relationships among features.

The analysis revealed a severe class imbalance: fraudulent transactions account for less than 0.2% of the original dataset. Consequently, accuracy alone is not sufficient for evaluating model performance.

### 2. Traditional Machine Learning Models

Three traditional classification approaches were evaluated:

- **Logistic Regression:** A linear classification baseline trained with balanced class weights.
- **Random Forest:** An ensemble of decision trees using balanced subsampling of classes.
- **Histogram-Based Gradient Boosting:** A boosting model trained with sample weights to account for class imbalance.

Random Forest was further optimized using `RandomizedSearchCV`, stratified two-fold cross-validation, and average precision as the model-selection metric. Four parameter configurations were evaluated.

The best configuration used:

- `n_estimators=100`
- `max_depth=None`
- `min_samples_leaf=2`
- `max_features='sqrt'`
- `class_weight='balanced_subsample'`

The best cross-validation average precision was approximately **0.8278**.

### 3. Deep Learning Model

A feed-forward neural network was implemented using TensorFlow and Keras.

The architecture consists of:

- An input layer accepting 30 features.
- A dense layer with 64 units and ReLU activation.
- Batch normalization and dropout with a rate of 0.3.
- A dense layer with 32 units and ReLU activation.
- Batch normalization and dropout with a rate of 0.2.
- An output layer with one unit and sigmoid activation.

The model was compiled using the Adam optimizer and binary cross-entropy loss. Evaluation metrics included ROC-AUC, PR-AUC, precision, and recall.

Training used TensorFlow's `tf.data` pipeline with batching, shuffling, and prefetching. Class weights addressed the imbalance between legitimate and fraudulent transactions.

Early stopping monitored validation PR-AUC, with a patience of five epochs and restoration of the best model weights. Training stopped after 18 epochs, with the best weights obtained at epoch 13.

### 4. Classification Threshold Optimization

The default classification threshold of 0.5 can be unsuitable for highly imbalanced datasets.

For each model, the validation predictions were used to investigate alternative thresholds. The selected threshold was the one with the highest precision among the evaluated thresholds that achieved at least 80% recall, where such a threshold was available.

For the tuned Random Forest, the selected threshold was approximately **0.5797**.

The test set was reserved for final evaluation. The threshold was selected using validation data and was not adjusted using test results.

## Results

### Validation Performance

The following table summarizes the validation results at the default threshold of 0.5.

| Model | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0611 | 0.9155 | 0.1145 | 0.9767 | 0.7217 |
| Random Forest baseline | 0.9355 | 0.8169 | 0.8722 | 0.9555 | 0.8593 |
| Histogram Gradient Boosting | 0.5495 | 0.8592 | 0.6703 | 0.9751 | 0.7442 |
| Neural Network | 0.0769 | 0.8732 | 0.1414 | 0.9681 | 0.7061 |
| Tuned Random Forest | 0.9355 | 0.8169 | 0.8722 | 0.9491 | 0.8618 |

The Random Forest models achieved the strongest balance of precision and recall at the default threshold. Logistic Regression and the neural network identified a high proportion of fraud cases but produced many false positives at that threshold.

### Threshold Optimization

Threshold selection improved the precision-recall trade-off for several models.

| Model | Selected Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9999 | 0.7532 | 0.8169 | 0.7838 |
| Random Forest baseline | 0.5931 | 0.9508 | 0.8169 | 0.8788 |
| Histogram Gradient Boosting | 0.9955 | 0.8636 | 0.8028 | 0.8321 |
| Neural Network | 0.9989 | 0.8169 | 0.8169 | 0.8169 |
| Tuned Random Forest | 0.5797 | 0.9667 | 0.8169 | 0.8855 |

The tuned Random Forest achieved the highest validation F1-score among the compared models after threshold optimization. Its validation precision was approximately 96.67%, with recall of 81.69%.

### Final Test Evaluation

The tuned Random Forest was selected for final evaluation using the threshold chosen on the validation set.

| Metric | Test Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 96.15% |
| Recall | 70.42% |
| F1-score | 0.8130 |
| ROC-AUC | 0.9415 |
| Average Precision | 0.8272 |

The test confusion matrix was:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 42,486 | 2 |
| Actual Fraud | 21 | 50 |

The model correctly identified 50 of the 71 fraudulent transactions in the test set while incorrectly flagging two legitimate transactions as fraudulent.

Although test accuracy was approximately 99.95%, this metric should be interpreted cautiously because legitimate transactions greatly outnumber fraudulent ones. Recall is particularly important because missed fraudulent transactions may result in financial losses.

Test recall was lower than validation recall, indicating that the model's performance did not transfer perfectly between the two sets. This difference is important when assessing the reliability of the final model.

## Discussion

The experiments demonstrate that model evaluation for fraud detection must account for the consequences of both false positives and false negatives.

Logistic Regression achieved a high ROC-AUC on the validation set but had low precision at the default threshold. This illustrates why a strong ranking metric does not necessarily translate into a useful classification decision at a particular threshold.

The Random Forest models provided a more balanced combination of precision and recall. Hyperparameter tuning produced a small improvement in validation average precision and, after threshold selection, a higher validation F1-score.

The neural network provided a useful deep learning comparison, but it did not outperform the tuned Random Forest on the validation F1-score. The neural network's threshold was also very close to 1, showing that its classification decisions were sensitive to the chosen operating point.

The final test results show that the tuned Random Forest can identify fraudulent transactions with high precision, but it still misses a meaningful proportion of fraud cases. In a real financial setting, the appropriate threshold would depend on the cost of missed fraud, the cost of investigating legitimate transactions, and operational requirements.

### Limitations

- The dataset contains very few fraudulent transactions compared with legitimate transactions.
- The anonymized PCA features limit direct interpretation of many predictors.
- The hyperparameter search was limited to four configurations and two-fold cross-validation.
- Validation and test performance differed, especially for recall.
- The experiment evaluates classification performance on a historical dataset and does not establish how the model would perform on future transactions or changing fraud patterns.
- No claim is made that the model is production-ready or has been tested in a live payment system.

## Conclusion

This project compared traditional machine learning models and a neural network for credit card fraud detection under severe class imbalance.

The tuned Random Forest achieved the strongest validation F1-score after threshold optimization. On the held-out test set, it achieved 96.15% precision, 70.42% recall, and an F1-score of 0.8130.

The results highlight the importance of stratified data splitting, imbalance-aware evaluation, model comparison, and validation-based threshold selection. They also demonstrate why high accuracy alone is insufficient for judging fraud detection systems.

Future work could explore broader hyperparameter searches, alternative imbalance-handling strategies, additional feature engineering, model interpretability, and evaluation on temporally separated data.
