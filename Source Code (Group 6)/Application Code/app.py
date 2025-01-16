import os
from flask import (Flask, redirect, render_template, request, send_from_directory, url_for, jsonify)
import utils as utils  # Import the utils module
from globals import chat_words, stopword, known_proper_nouns, spell  # Import the global variables

app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('interface.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    text = data['text']
    model_name = data.get('model', 'bert')  # Change model here

    print(f"Analyzing text: {text}")
    sentiment, keywords, confidence_score = utils.load_and_predict(text, model_name)

    return jsonify(sentiment=sentiment, keywords=keywords, confidence_score=confidence_score)

if __name__ == '__main__':
   app.run('0.0.0.0', port='8080')
