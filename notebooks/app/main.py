# app/main.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# 🔄 Load models
model = joblib.load("model/stacking_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")
tfidf_vectorizers = joblib.load("model/tfidf_vectorizers.pkl")

# ✨ Create Flask app
app = Flask(__name__)

# 📋 Define columns
text_cols = list(tfidf_vectorizers.keys())



@app.route("/")
def home():
    return "✅ Fraud Detection API is up!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])  # Convert to DataFrame

        # -------------------------------
        # ✅ Feature Engineering
        # -------------------------------
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob

        analyzer = SentimentIntensityAnalyzer()

        # Fill missing text columns with empty strings
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""

        # Create `all_text` column
        df["all_text"] = df["title"] + " " + df["description"] + " " + \
                         df.get("requirements", "") + " " + \
                         df.get("company_profile", "") + " " + \
                         df.get("benefits", "")

        # Sentiment analysis (VADER)
        vader_scores = df["all_text"].apply(lambda x: analyzer.polarity_scores(x))
        df["vader_neg"] = vader_scores.apply(lambda x: x["neg"])
        df["vader_neu"] = vader_scores.apply(lambda x: x["neu"])
        df["vader_pos"] = vader_scores.apply(lambda x: x["pos"])
        df["vader_compound"] = vader_scores.apply(lambda x: x["compound"])

        # TextBlob polarity
        df["textblob_polarity"] = df["all_text"].apply(lambda x: TextBlob(x).sentiment.polarity)

        # Keyword detection
        keywords = ["money", "click", "investment", "earn", "urgent", "opportunity", "work from home"]
        for kw in keywords:
            colname = f"keyword_{kw}"
            df[colname] = df["all_text"].str.lower().str.contains(kw).astype(int)

        # -------------------------------
        # ✅ Continue as before
        # -------------------------------
        # 🧩 Handle missing categorical columns
        required_columns = [
            'required_education', 'department', 'function',
            'employment_type', 'industry', 'required_experience'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        non_text_cols = [col for col in df.columns if col not in text_cols]

        # Process tabular
        X_tabular = preprocessor.transform(df[non_text_cols])

        # Process text
        text_vectors = []
        for col in text_cols:
            tfidf = tfidf_vectorizers[col]
            vec = tfidf.transform(df[col].fillna(""))
            text_vectors.append(vec)

        X_text = hstack(text_vectors)
        X_final = hstack([csr_matrix(X_tabular), X_text])

        # Predict
        pred = model.predict(X_final)[0]
        prob = model.predict_proba(X_final)[0][1]

        result = {
            "prediction": "Fraudulent" if pred == 1 else "Legit",
            "fraud_probability": round(prob, 3)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

from PIL import Image
import pytesseract
import io

@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        print("Request files:", request.files)  # DEBUG
        if 'file' not in request.files:
            return jsonify({"error": "No image file provided"})

        file = request.files['file']
        # rest of your code...

        image = Image.open(file.stream)

        # 🔍 Extract text from image
        extracted_text = pytesseract.image_to_string(image)
        print("FILES RECEIVED:", request.files)


        # 🔄 Convert to DataFrame with minimal required structure
        df = pd.DataFrame([{
            "title": "",
            "description": extracted_text,
            "company_profile": "",
            "requirements": "",
            "benefits": "",
            "required_education": "",
            "department": "",
            "function": "",
            "employment_type": "",
            "industry": "",
            "required_experience": ""
        }])

        # 🔁 Duplicate your existing feature engineering pipeline
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob

        analyzer = SentimentIntensityAnalyzer()

        df["all_text"] = df["title"] + " " + df["description"] + " " + \
                         df.get("requirements", "") + " " + \
                         df.get("company_profile", "") + " " + \
                         df.get("benefits", "")

        # VADER
        vader_scores = df["all_text"].apply(lambda x: analyzer.polarity_scores(x))
        df["vader_neg"] = vader_scores.apply(lambda x: x["neg"])
        df["vader_neu"] = vader_scores.apply(lambda x: x["neu"])
        df["vader_pos"] = vader_scores.apply(lambda x: x["pos"])
        df["vader_compound"] = vader_scores.apply(lambda x: x["compound"])

        # TextBlob
        df["textblob_polarity"] = df["all_text"].apply(lambda x: TextBlob(x).sentiment.polarity)

        # Keywords
        keywords = ["money", "click", "investment", "earn", "urgent", "opportunity", "work from home"]
        for kw in keywords:
            colname = f"keyword_{kw}"
            df[colname] = df["all_text"].str.lower().str.contains(kw).astype(int)

        # Handle missing text cols
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""

        non_text_cols = [col for col in df.columns if col not in text_cols]
        X_tabular = preprocessor.transform(df[non_text_cols])

        text_vectors = []
        for col in text_cols:
            tfidf = tfidf_vectorizers[col]
            vec = tfidf.transform(df[col].fillna(""))
            text_vectors.append(vec)

        X_text = hstack(text_vectors)
        X_final = hstack([csr_matrix(X_tabular), X_text])

        pred = model.predict(X_final)[0]
        prob = model.predict_proba(X_final)[0][1]

        result = {
            "prediction": "Fraudulent" if pred == 1 else "Legit",
            "fraud_probability": round(prob, 3),
            "extracted_text": extracted_text[:500] + "..."  # limit preview
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# 🏁 Run server
if __name__ == "__main__":
    app.run(debug=True)
