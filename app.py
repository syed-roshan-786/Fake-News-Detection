from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import os
from utils.preprocessing import clean_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# MySQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Roshan%40123@localhost/fake_news_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load models
svm_model = joblib.load(os.path.join(BASE, 'models', 'svm_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE, 'models', 'vectorizer.pkl'))

# Database model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(16), nullable=False)
    prob_real = db.Column(db.Float, nullable=True)
    prob_fake = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'input_text': self.input_text,
            'prediction': self.prediction,
            'prob_real': self.prob_real,
            'prob_fake': self.prob_fake,
            'timestamp': self.timestamp.isoformat()
        }

with app.app_context():
    db.create_all()

# Utility: Generate WordCloud image from recent news
def generate_wordcloud():
    recent_news = History.query.order_by(History.timestamp.desc()).limit(100).all()
    if not recent_news:
        return  # Skip if no news

    text_data = " ".join([clean_text(r.input_text) for r in recent_news])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    # Save image to static folder
    wordcloud_path = os.path.join(BASE, 'static', 'wordcloud.png')
    wordcloud.to_file(wordcloud_path)

# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form.get("news", "")
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        pred = svm_model.predict(features)[0]
        prob_real = prob_fake = None
        try:
            probs = svm_model.predict_proba(features)[0]
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
        except Exception:
            pass

        label = "REAL" if pred == 1 else "FAKE"

        # Save prediction to database
        rec = History(input_text=text[:5000], prediction=label,
                      prob_real=prob_real, prob_fake=prob_fake)
        db.session.add(rec)
        db.session.commit()

        result = {'label': label, 'prob_real': prob_real, 'prob_fake': prob_fake, 'id': rec.id}

    # Generate WordCloud for homepage
    generate_wordcloud()

    # Fetch recent predictions
    recent = History.query.order_by(History.timestamp.desc()).limit(10).all()
    return render_template("index_boot.html", result=result, recent=recent)

# API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    text = data.get("text", "")
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = svm_model.predict(features)[0]
    prob_real = prob_fake = None
    try:
        probs = svm_model.predict_proba(features)[0]
        prob_fake, prob_real = float(probs[0]), float(probs[1])
    except Exception:
        pass

    label = "REAL" if pred == 1 else "FAKE"
    rec = History(input_text=text[:5000], prediction=label,
                  prob_real=prob_real, prob_fake=prob_fake)
    db.session.add(rec)
    db.session.commit()

    # Update WordCloud whenever a new prediction is added
    generate_wordcloud()

    return jsonify({"prediction": label, "prob_real": prob_real, "prob_fake": prob_fake, "id": rec.id})

# History page
@app.route("/history")
def history_page():
    rows = History.query.order_by(History.timestamp.desc()).all()
    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
