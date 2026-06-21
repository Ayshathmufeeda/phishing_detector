import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

# 1. Create a sample dataset - replace with your own CSV later
data = {
    'email_text': [
        "Dear user, verify your account now at http://secure-bank-login.com",
        "Hi John, meeting at 3pm tomorrow. Regards, Sarah",
        "URGENT: Your PayPal account suspended. Click here: http://bit.ly/2xYz",
        "Invoice for your recent purchase attached. Thank you",
        "Congrats! You won $1000. Claim prize http://win-money-now.net",
        "Project files for review. Let me know your thoughts",
        "Security alert: Unusual login. Verify identity http://accounts-google.co",
        "Happy birthday! Hope you have a great day",
        "Update your password immediately http://amazon-security.org/login",
        "Can you send the report by EOD? Thanks"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1 = Phishing, 0 = Safe
}
df = pd.DataFrame(data)

# 2. Feature Extraction: URLs, keywords, etc.
def extract_features(text):
    # Count URLs
    url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    # Check for phishing keywords
    keywords = ['urgent', 'verify', 'suspended', 'claim', 'prize', 'password', 'security', 'alert', 'click']
    keyword_count = sum(1 for word in keywords if word in text.lower())
    return f"{text} URLCOUNT_{url_count} KEYWORDCOUNT_{keyword_count}"

df['processed_text'] = df['email_text'].apply(extract_features)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label'], test_size=0.3, random_state=42
)

# 4. Vectorize text + Train model
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== Phishing Email Detection Results ===")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(" Predicted Safe Predicted Phishing")
print(f"Actual Safe {cm[0][0]} {cm[0][1]}")
print(f"Actual Phishing {cm[1][0]} {cm[1][1]}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# 6. Test on new email
def predict_email(text):
    processed = extract_features(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "Phishing" if prediction == 1 else "Safe"

print("\n=== Testing New Emails ===")
test_email1 = "Final notice: Your account will be closed. Login http://fake-bank.com"
test_email2 = "Hey, are we still on for lunch today?"
print(f"Email: '{test_email1}' -> {predict_email(test_email1)}")
print(f"Email: '{test_email2}' -> {predict_email(test_email2)}")