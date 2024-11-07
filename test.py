import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os 
import pickle

#-----------------------------------load the model and saved tokenizers, max lengths------------------------
# Define the model directory
model_dir = './LSTM_select_glove'

# Load the saved model
model = tf.keras.models.load_model(model_dir)
print(model)


# Load tokenizers
with open(os.path.join(model_dir, 'question_tokenizer.pkl'), 'rb') as f:
    question_tokenizer = pickle.load(f)
with open(os.path.join(model_dir, 'query_tokenizer.pkl'), 'rb') as f:
    query_tokenizer = pickle.load(f)

# Load max lengths
with open(os.path.join(model_dir, 'max_lengths.pkl'), 'rb') as f:
    lengths = pickle.load(f)
    max_question_len = lengths['max_question_len']
    max_query_len = lengths['max_query_len']



#--------load then model when not using saved tokenizer, have to tokenize for output by loading dataset------

# # Load your data (same data used for training)
# data = pd.read_csv('./database/train_own_small.csv')
# questions = data['question'].values
# queries = data['query'].values

# # Load the model
# model = tf.keras.models.load_model('./trained 10')

# # Initialize tokenizer for questions and queries
# question_tokenizer = Tokenizer()
# query_tokenizer = Tokenizer(filters='')

# # Fit the tokenizer on the questions and queries
# question_tokenizer.fit_on_texts(questions)
# query_tokenizer.fit_on_texts(queries)

# # Determine max lengths
# max_question_len = max(len(seq) for seq in question_tokenizer.texts_to_sequences(questions))
# max_query_len = max(len(seq) for seq in query_tokenizer.texts_to_sequences(queries))


def predict_query(question):
    # Tokenize and pad the input question
    sequence = question_tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=max_question_len, padding='post')
    
    # Make prediction   
    prediction = model.predict(padded_sequence)
    
    # # Debug: Check the shape and content of prediction
    # print("Prediction shape:", prediction.shape)
    # print("Prediction output:", prediction)
    
    # Convert the prediction to text
    predicted_sequence = np.argmax(prediction, axis=-1)
    predicted_query = query_tokenizer.sequences_to_texts(predicted_sequence)

    # Strip newline characters and other unwanted characters
    cleaned_query = [query.replace('\n', ' ').replace('[',' ').replace(']',' ').strip() for query in predicted_query]
    
    return cleaned_query[0] if cleaned_query else ''
    


test_questions = [

# Place your test questions here for predictions
   

]
def load_questions(file_path):
    with open(file_path, 'r') as file:
        questions = file.readlines()
    return questions
        
file_path = './select2.txt'
questions = load_questions(file_path)

for test_question in questions:
    test_question = test_question.strip()
    predicted_query = predict_query(test_question)
    print(f"Question: {test_question}")
    print(f"Predicted Query: {predicted_query}")
    print()

print("***************************************************************************************************")
print(f'\n using model {model_dir} \n ')










