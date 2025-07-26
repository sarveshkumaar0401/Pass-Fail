# Pass-Fail
"""
# 📊 Student Exam Pass/Fail Prediction

This project uses **Logistic Regression** to predict whether a student will **pass** or **fail** an exam based on:

- **Study Hours**
- **Previous Exam Score**

## 🗂️ Dataset

The dataset used: `student_exam_data_new.csv`  
It should contain the following columns:
- `Study Hours` (numeric)
- `Previous Exam Score` (numeric)
- `Pass/Fail` (target variable, categorical: "Pass" or "Fail")

## 🛠️ Requirements

- Python 3.x  
- pandas  
- scikit-learn  

Install the requirements using:


## 🧠 Model Pipeline

1. Data Loading using pandas
2. Feature Selection: Study Hours and Previous Exam Score
3. Label Encoding the target (Pass/Fail)
4. Train-Test Split (80/20)
5. Model Training using LogisticRegression
6. Prediction on new data samples
7. Output: Probability & Prediction (Pass/Fail)

## 🔍 Sample Prediction

Sample inputs:
```python
new_input = pd.DataFrame({
    'Study Hours' : [10 ,9],
    'Previous Exam Score' :[93,43]
})


   Probability of Passing Prediction
0                     1.0       Pass
1                     0.0       Fail
