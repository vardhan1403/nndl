import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import io
import sys

# Set the encoding of stdout to utf-8 to handle special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
text = "".join(dataset["train"]["text"][:1000])  # First 1000 lines for testing

# Tokenize and limit vocabulary size
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Prepare input sequences
input_sequences = []
for line in text.split("."):  # Split by periods to process sentence-like structures
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform input length
max_sequence_len = 20
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split predictors and labels
predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define the LSTM model
model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_len - 1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model (we are using fewer epochs for testing)
history = model.fit(predictors, labels, epochs=10, verbose=1)

# Function to generate the next word prediction
def generate_next_word(model, tokenizer, input_text, max_sequence_len=20):
    # Tokenize the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]

    # If the sequence is too long, truncate it to fit the model input
    if len(input_sequence) > max_sequence_len - 1:
        input_sequence = input_sequence[-(max_sequence_len - 1):]

    # Pad the sequence if it's shorter than the required length
    input_sequence = np.pad(input_sequence, (max_sequence_len - 1 - len(input_sequence), 0), mode='constant')

    # Reshape the input sequence for the LSTM model
    input_sequence = np.array(input_sequence).reshape(1, max_sequence_len - 1)

    # Make the prediction
    prediction = model.predict(input_sequence)

    # Get the predicted word index
    predicted_index = np.argmax(prediction)

    # Convert index to word
    predicted_word = tokenizer.index_word[predicted_index]

    return predicted_word

# Example usage
input_text = "The quick brown fox"
predicted_word = generate_next_word(model, tokenizer, input_text)

# Write the output to a file (to avoid console encoding issues)
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(f"Input: {input_text}\n")
    file.write(f"Predicted next word: {predicted_word}\n")

print("Output written to output.txt")
