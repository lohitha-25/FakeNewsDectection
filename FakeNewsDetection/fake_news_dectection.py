import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1. Load a larger dataset (you can replace this with an actual dataset, e.g., Kaggle's fake news dataset)
data = {
    'text': [
        'Donald Trump sends out tweets about fake news.',
        'NASA finds water on the moon.',
        'Aliens have landed on Earth!',
        'Scientists discover a cure for cancer.',
        'President caught in major corruption scandal.',
        'The earth is flat and NASA lies to us.',
        'Breaking news: The stock market is crashing.',
        'Global warming is the greatest threat to humanity.',
        'New study shows a link between sugar and cancer.',
        'New tech company innovates in the AI field.'
    ],
    'label': [
        'FAKE', 'REAL', 'FAKE', 'REAL', 'REAL', 'FAKE', 'REAL', 'REAL', 'FAKE', 'REAL'
    ]
}

df = pd.DataFrame(data)

# 2. Preprocessing
X = df['text']
y = df['label']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Create models
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100)

# 6. Use Voting Classifier (Ensemble method)
ensemble_model = VotingClassifier(estimators=[('lr', log_reg), ('rf', random_forest)], voting='hard')

# 7. Hyperparameter Tuning using GridSearchCV for Logistic Regression
param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 200, 300]}
grid_search = GridSearchCV(log_reg, param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Best parameters from GridSearchCV
print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")

# Use best parameters for Logistic Regression
best_log_reg = grid_search.best_estimator_

# 8. Train ensemble model
ensemble_model.fit(X_train_tfidf, y_train)

# 9. Evaluate ensemble model
y_pred = ensemble_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Ensemble Model Accuracy: {accuracy*100:.2f}%")

# 10. Display Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 12. Plot a Bar Graph for Accuracy and Other Metrics
metrics = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(metrics).transpose()

# Plot the Bar Graph
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 5))
plt.title('Precision, Recall, F1-Score Comparison')
plt.ylabel('Score')
plt.xlabel('Metrics')
plt.xticks(rotation=0)
plt.show()

# 13. Save the model and vectorizer for later use
joblib.dump(ensemble_model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# 14. Predict user input
while True:
    user_input = input("\nEnter news text (or type 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        break
    input_vec = vectorizer.transform([user_input])
    prediction = ensemble_model.predict(input_vec)[0]
    print(f"🧠 Prediction: This news is *{prediction.upper()}*.")
