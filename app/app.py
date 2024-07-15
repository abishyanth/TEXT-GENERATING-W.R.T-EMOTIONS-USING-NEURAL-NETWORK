from flask import Flask, request, render_template
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle

app = Flask(__name__)

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
characters = sorted(set(text))

# Load model and necessary components
with open('./shakespeare_model.pkl', 'rb') as f:
    model_config, model_weights, char_to_index, index_to_char = pickle.load(f)

model = model_from_json(model_config)
model.set_weights(model_weights)
SEQ_LENGTH = 40

# Function to sample predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

# Function to generate mood-based text
def generate_mood_text(mood, length, temperature):
    mood_dict = {
        "happy": ["joyful", "cheerful", "optimistic", "excited"],
        "sad": ["sorrowful", "depressed", "melancholy", "tearful"],
        "angry": ["furious", "enraged", "indignant", "hostile"],
        "calm": ["peaceful", "serene", "tranquil", "composed"],
    }

    mood_words = mood_dict.get(mood.lower(), [])

    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]

        filtered_predictions = [index for index, prediction in enumerate(predictions) if index_to_char[index] in mood_words]

        if len(filtered_predictions) > 0:
            next_index = random.choice(filtered_predictions)
        else:
            next_index = sample(predictions, temperature)

        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_generation')
def text_generation():
    return render_template('text_generation.html')

@app.route('/generate_mood_text', methods=['POST'])
def generate_mood_text_route():
    mood = request.form['mood']
    length = int(request.form['length'])
    temperature = float(request.form['temperature'])
    generated_text = generate_mood_text(mood, length, temperature)
    return render_template('text_generation.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
