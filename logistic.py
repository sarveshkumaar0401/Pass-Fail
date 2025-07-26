import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("D:\ML\logistic\student_exam_data_new.csv")

print(df.head())

X=df[['Study Hours','Previous Exam Score']]
y=df['Pass/Fail']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

new_input = pd.DataFrame({
    'Study Hours' : [10 ,9],
    'Previous Exam Score' :[93,43]
})
probs = model.predict(new_input)

rounded_preds = [1 if p >= 0.5 else 0 for p in probs]
predicted_labels = ['Pass' if val == 1 else 'Fail' for val in rounded_preds]


results_df = pd.DataFrame({
    'Probability of Passing': probs.round(2),
    'Prediction': predicted_labels
})


# Evaluate on test set
#y_test_probs = model.predict_proba(X_test)[:, 1]
#y_test_preds = [1 if p >= 0.5 else 0 for p in y_test_probs]

#print("Accuracy:", accuracy_score(y_test, y_test_preds))
#print("\nClassification Report:\n", classification_report(y_test, y_test_preds, target_names=['Fail', 'Pass']))


print("\nSample Prediction:\n")
print(results_df)