import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Download stopwords for text cleaning
nltk.download("stopwords")

# Function to clean text
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\d+", " ", text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = " ".join(word for word in text.split() if word not in stopwords.words("english"))
    return text

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Check dataset structure
print(df.head())

# Ensure columns are named correctly
if "Resume" not in df.columns or "Category" not in df.columns:
    raise ValueError("CSV file must contain 'Resume' and 'Category' columns!")

# Clean resume text
df["cleaned_resume"] = df["Resume"].apply(clean_text)

# Encode categories
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["Category"])

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["cleaned_resume"]).toarray()
y = df["category_encoded"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save the trained model, vectorizer, and encoder
with open("clf.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model training complete! Files saved: clf.pkl, tfidf.pkl, encoder.pkl")
