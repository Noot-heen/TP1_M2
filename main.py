from flask import Flask, render_template, request
import joblib
import nltk, re
from nltk.corpus import stopwords

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')

# Charge le modèle et le vectoriseur
model = joblib.load('model/spam_model_fr.pkl')
vectorizer = joblib.load('model/vectorizer_fr.pkl')

# Stop words français (comme pendant l'entraînement)
stop_words_fr = set(stopwords.words('french'))

def preprocess_fr(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words_fr]
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def main():
    prediction = None
    confidence = None
    message = None
    seuil = 0.5  # Valeur par défaut
        
    if request.method == 'POST':
        message = request.form.get('message')
        print(message)
        seuil_str = request.form.get('seuil', '0.5')
        try:
            seuil = float(seuil_str)
            if not 0.0 <= seuil <= 1.0:
                seuil = 0.5
        except:
            seuil = 0.5

        if message:
            print(message)
            cleaned = preprocess_fr(message)
            vec = vectorizer.transform([cleaned])
            proba = model.predict_proba(vec)[0][1]  # Probabilité SPAM
            result = "SPAM" if proba >= seuil else "HAM"
            prediction = result
            confidence = round(proba * 100, 2)  # En %

    return render_template('index.html',
                            prediction=prediction,
                            confidence=confidence,
                            message=message,
                            seuil=seuil
                        )
