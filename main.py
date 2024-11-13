import json
from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from LemmaTokenizer import LemmaTokenizer

app = Flask(__name__)
CORS(app)

def get_recommendation(process):
    return recommendations.get(process, "Нет доступных рекомендаций для данного процесса.")


@app.route('/predict', methods=['GET'])
def predict():
    log_message = request.args['log_message']
    log_vector = pipeline.transform([log_message]).astype('float32')
    prediction = model.predict(log_vector)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]
    recommendation = get_recommendation(predicted_label)
    response = {
        "Predicted process": predicted_label,
        "Reccomendation": recommendation
    }
    return jsonify(response)


if __name__ == '__main__':
    port = 7777
    with open("./model/recommendations.json", "rb") as f4:
        recommendations = json.load(f4)
    f4.close()

    nltk.download('wordnet')
    nltk.download('omw-1.4')

    model = joblib.load("./model/model.pkl")
    pipeline = joblib.load("./model/pipeline.pkl")
    encoder = joblib.load("./model/encoder.pkl")

    app.run(host='0.0.0.0', port=port, debug=True)
