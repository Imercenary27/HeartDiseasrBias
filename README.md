# Gender Bias Analysis in Heart Disease Prediction

This project analyzes and mitigates gender bias in a machine learning model designed to predict heart disease. The primary goal is to build a fair AI model that delivers equitable healthcare predictions, aligning with the UN's Sustainable Development Goal 3 (Good Health and Well-being) by addressing health disparities.

The analysis uses the UCI Heart Disease dataset to train a logistic regression model, evaluates it for fairness, and implements several bias mitigation techniques to reduce discrimination based on gender.

## Dataset

The project utilizes the **Heart Disease UCI** dataset, specifically the Cleveland subset, which contains 14 attributes, including the target variable indicating the presence of heart disease. The analysis focuses on gender (`sex`) as the protected attribute to evaluate model bias.

## Methodology

The project follows a structured approach to identify and mitigate algorithmic bias.

### 1. Baseline Model

A baseline **Logistic Regression** model is first trained on the dataset without any modifications. This model's performance and fairness metrics serve as a benchmark to evaluate the effectiveness of mitigation techniques.

### 2. Bias Identification

Bias is identified by evaluating the baseline model's predictions against key fairness metrics:
*   **Demographic Parity Difference:** Measures the difference in selection rates (the rate at which the model predicts a positive outcome) between males and females. A value of 0 indicates perfect fairness.
*   **Equalized Odds Difference:** Measures the difference in true positive rates between gender groups. This metric helps ensure the model performs equally well for both men and women who actually have heart disease.

### 3. Bias Mitigation Techniques

Three pre-processing and in-processing techniques were implemented to reduce the identified bias:

*   **SMOTE (Synthetic Minority Over-sampling Technique):** A data-level method that oversamples the minority group in the training data to create a more balanced dataset before training the model.
*   **Class Weight Balancing:** An in-processing technique where the `LogisticRegression` algorithm is configured to give higher weight to minority samples during training, penalizing misclassifications of the underrepresented group more heavily.
*   **Reweighting:** A pre-processing technique that assigns weights to samples in the dataset to ensure fairness. This implementation is based on the methodology proposed by Kamiran and Calders.

## Results

The performance and fairness of the baseline model were compared against the models trained with bias mitigation techniques. The results are summarized in the table below, with the best-performing method highlighted for its ability to reduce bias while maintaining accuracy.

| Method | Accuracy | Demographic Parity Difference (DPD) | Equalized Odds Difference (EOD) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 81.8% | 0.221 | 0.111 |
| **SMOTE** | 83.1% | 0.183 | 0.108 |
| **Balanced** | 84.4% | 0.202 | 0.108 |

Based on the analysis, **SMOTE** was identified as the most effective method, as it achieved the lowest Demographic Parity Difference, indicating a significant reduction in bias, along with a slight improvement in overall accuracy.

## How to Run

1.  Ensure all required dependencies are installed.
2.  Place the `heart_disease_uci.csv` file in the same directory as the script.
3.  Run the main analysis script from your terminal:
    ```
    python caprt2.py
    ```
4.  The script will execute the complete bias analysis pipeline, print the results for each model, and save a summary to `heart_disease_bias_results.csv`.

## Dependencies

This project requires the following Python libraries:
*   pandas
*   NumPy
*   scikit-learn
*   fairlearn
*   imblearn
