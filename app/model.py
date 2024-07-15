import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Function to preprocess text and train the model
# Function to create and train the model
filepath = tf.keras.utils.get_file('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8')
text = text[30000:800000]

def train_model(text):
    text = text.lower()
    characters = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(characters)}
    index_to_char = {i: c for i, c in enumerate(characters)}

    SEQ_LENGTH = 40
    STEP_SIZE = 3
    sentences = []
    next_char = []

    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i: i + SEQ_LENGTH])
        next_char.append(text[i + SEQ_LENGTH])

    x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
    y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

    for i, satz in enumerate(sentences):
        for t, char in enumerate(satz):
            x[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_char[i]]] = 1

    model = Sequential([
        LSTM(128, input_shape=(SEQ_LENGTH, len(characters))),
        Dense(len(characters)),
        Activation('softmax')
    ])
    
    optimizer = RMSprop(learning_rate=0.01)  # Specify learning rate this way

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit(x, y, batch_size=256, epochs=4, verbose=1)
    
    # Save model and related components using pickle
    with open('shakespeare_model.pkl', 'wb') as f:
        pickle.dump((model.to_json(), model.get_weights(), char_to_index, index_to_char), f)

    return model, char_to_index, index_to_char

# Function to load the model and components from pickle file
def load_model_and_components():
    with open('shakespeare_model.pkl', 'rb') as f:
        model_json, model_weights, char_to_index, index_to_char = pickle.load(f)
    
    model = tf.keras.models.model_from_json(model_json)
    model.set_weights(model_weights)
    
    return model, char_to_index, index_to_char

# Function to generate text
def generate_text(model, text, char_to_index, index_to_char, length, temperature):
    SEQ_LENGTH = 40  # Assuming SEQ_LENGTH is known and consistent
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for _ in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(char_to_index)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated

# Function to sample predictions
def sample(preds, temperature=100):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# # Example usage
# if __name__ == "__main__":
#     # Train the model
#     model, char_to_index, index_to_char = train_model(text)
