import tensorflow as tf

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used.")
else:
    print("No GPU found, using CPU instead.")
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector, TimeDistributed, GlobalMaxPooling1D, Dropout
import pandas as pd
import pickle
import os


# Load your data
data = pd.read_csv('./database/train_new.csv')
questions = data['question'].values
queries = data['query'].values

# Initialize tokenizer for questions and queries
question_tokenizer = Tokenizer()
query_tokenizer = Tokenizer(filters='')

# Fit the tokenizer on the questions and queries
question_tokenizer.fit_on_texts(questions)
query_tokenizer.fit_on_texts(queries)

# Convert texts to sequences
questions_seq = question_tokenizer.texts_to_sequences(questions)
queries_seq = query_tokenizer.texts_to_sequences(queries)

print(questions_seq)
print(queries_seq)

# Pad sequences
max_question_len = max(len(seq) for seq in questions_seq)
max_query_len = max(len(seq) for seq in queries_seq)
questions_padded = pad_sequences(questions_seq, maxlen=max_question_len, padding='post')
queries_padded = pad_sequences(queries_seq, maxlen=max_query_len, padding='post')

# Convert data to numpy arrays
X = np.array(questions_padded)
y = np.array(queries_padded)



# Assuming X and y are your input features and target labels, respectively

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=32)

#GLove
# You can also skip using glove embedding based on your model's results
embedding_dim = 300
glove_path = './glove.840B.300d/glove.840B.300d.txt' # path to your glove embeddings

# Create an embedding matrix 
vocab_size_questions = len(question_tokenizer.word_index) + 1
vocab_size_queries = len(query_tokenizer.word_index) + 1

embedding_matrix = np.zeros((vocab_size_questions, embedding_dim))
#sperate matrix for question and queries


with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split(' ')
        word = values[0]

        # Try to convert the rest of the values to floats
        try:
            vector = np.asarray(values[1:], dtype='float32')
        except ValueError as e:
            print(f"Skipping line due to error: {e}")
            continue

        # Check if the word is in the tokenizer's word index
        if word in question_tokenizer.word_index:
            index = question_tokenizer.word_index[word]
            embedding_matrix[index] = vector


print(vocab_size_questions)
print(vocab_size_queries)


# Build the model
model = Sequential()
model.add(Embedding(vocab_size_questions, embedding_dim,weights=[embedding_matrix], input_length=max_question_len))
# model.add(LSTM(250, return_sequences=True)) #new
# model.add(Dropout(0.1))                     #new
model.add(LSTM(250, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(RepeatVector(max_query_len))
model.add(LSTM(250, return_sequences=True))
# model.add(Dropout(0.1))                     #new
model.add(TimeDistributed(Dense(vocab_size_queries, activation='softmax')))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

my_model = model.fit(X_train,y_train, epochs=250, batch_size=32, validation_split = 0.1)

#------------------------------------------OR--------------------------------------------------
# model.add(LSTM(250, return_sequences=True))
# model.add(LSTM(250, return_sequences=True))
# model.add(GlobalMaxPooling1D())
# model.add(RepeatVector(max_query_len))


loss, accuracy = model.evaluate(X_test,y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

def predict_query(question):
    sequence = question_tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=max_question_len, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_sequence = np.argmax(prediction, axis=-1)
    return query_tokenizer.sequences_to_texts(predicted_sequence)

# Test prediction
test_question = "which is the cheapest bike"

# -------------------------------------------------save only the model--------------------------------------
# predicted_query = predict_query(test_question)
# print(predicted_query)

# model.save('./trained 10')



predicted_query = predict_query(test_question)
print(predicted_query)

#-------------------------------------save the model,tokenizers,max lengths-------------------------------

# Define the model directory
model_dir = './LSTM_order_glove'

# Create the directory if it does not exist
os.makedirs(model_dir, exist_ok=True)

# Save the model in TensorFlow SavedModel format
model.save(model_dir)

# Save tokenizers
with open(os.path.join(model_dir, 'question_tokenizer.pkl'), 'wb') as f:
    pickle.dump(question_tokenizer, f)
with open(os.path.join(model_dir, 'query_tokenizer.pkl'), 'wb') as f:
    pickle.dump(query_tokenizer, f)

# Save max lengths
with open(os.path.join(model_dir, 'max_lengths.pkl'), 'wb') as f:
    pickle.dump({'max_question_len': max_question_len, 'max_query_len': max_query_len}, f)
