import re
import streamlit as st
import mysql.connector
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
import google.generativeai as genai
import speech_recognition as sr

# Prompt for Gemini model
prompt = [
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the 4 Table named - bikes, customers, dealers, purchases
    (details omitted for brevity)
    """
]

# Function to connect to MySQL
def connect_to_database():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='yash@12345',
        database='office'
    )

# Function To Load Google Gemini Model and provide queries as response
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([prompt[0], question])
    cleaned_response = response.text.replace("```sql", "").replace("```", "").strip()
    return cleaned_response

# Load the main model
def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    
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

    return model, question_tokenizer, query_tokenizer, max_question_len, max_query_len

def predict_query(model, question_tokenizer, query_tokenizer, max_question_len, question):
    # Tokenize and pad the input question
    sequence = question_tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=max_question_len, padding='post')
        
    # Make prediction   
    prediction = model.predict(padded_sequence)
        
    # Convert the prediction to text
    predicted_sequence = np.argmax(prediction, axis=-1)
    predicted_query = query_tokenizer.sequences_to_texts(predicted_sequence)

    # Strip newline characters and other unwanted characters
    cleaned_query = [query.replace('\n', ' ').replace('[',' ').replace(']',' ').strip() for query in predicted_query]
        
    return cleaned_query[0] if cleaned_query else ''



# Function to execute SQL query
def execute_query(sql_query):
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return None

# Function to handle speech-to-text using SpeechRecognition
def record_text():
    reco = sr.Recognizer()
    with sr.Microphone() as source:
        info_message = st.info("Listening for audio input...")
        reco.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = reco.listen(source, timeout=6, phrase_time_limit=6)
            MyText = reco.recognize_google(audio)  # or use recognize_sphinx for offline
            st.success(f"Recognized Speech: {MyText}")
            info_message.empty()
            st.info("Recognition ended.....")
            return MyText
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except sr.UnknownValueError:
            st.error("Speech not recognized or unknown error.")
        return ""

# Main app
def main():
    st.title("Dialect DB")
    # Use st.markdown with custom CSS for smaller text
    st.markdown("<h3 style='font-size:17px;'> A RNN based Text to SQL Model</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize session state to track query execution
    if 'query_executed' not in st.session_state:
        st.session_state.query_executed = False

    # Text input
    nlp_input_text = st.text_input("Enter your query:")
    
    sql_query = ""
    
    # Option to choose input method
    input_method = st.radio("Choose Input Method", ("Text Input", "Speech Input"))

    if input_method == "Speech Input":
        if st.button("Start Recording"):
            nlp_input_text = record_text()
            # st.write(f"Recognized Query Text: {nlp_input_text}")
            model_dir_main = './LSTM_trained_pooling5_without_glove'
            model_main, question_tokenizer_main, query_tokenizer_main, max_question_len_main, _ = load_model(model_dir_main)
        
            # Predict SQL query using the main model
            sql_query_main = predict_query(model_main, question_tokenizer_main, query_tokenizer_main, max_question_len_main, nlp_input_text)
            
            # Check for the appropriate model based on output generated by the main model
            match_select = re.search(r'\bselect\b', sql_query_main, re.IGNORECASE)
            match_order = re.search(r'\border by\b', sql_query_main, re.IGNORECASE)
            match_join = re.search(r'\bjoin\b', sql_query_main, re.IGNORECASE)
            
            if match_join:
                model_dir = './LSTM_join_glove'
            elif match_order:
                model_dir = './LSTM_order_glove' 
            elif match_select:
                model_dir = './LSTM_select_glove'
            else:
                st.write("Can't load model, check again.")
                return
            
            # Load the appropriate model
            model, question_tokenizer, query_tokenizer, max_question_len, _ = load_model(model_dir)
            st.write(f"using model{model_dir}")
            
            # Predict with the specific model and display result
            sql_query = predict_query(model, question_tokenizer, query_tokenizer, max_question_len, nlp_input_text)
            st.subheader("Generated SQL Query: ")
            st.code(sql_query)
            st.session_state.sql_query = sql_query
            
            # Execute query
            query_result = execute_query(sql_query)

            # Display results
            if query_result:
                st.write("Query Results:")
                for row in query_result:
                    st.write(row)
            else:
                st.write("No results found.")

            # Update session state to indicate query execution
            st.session_state.query_executed = True
            
    
            
            



    # Show "Generate" button only if the query has not been executed
    if not st.session_state.query_executed:
        if st.button("Generate"):
            # Load main model
            model_dir_main = './LSTM_trained_pooling5_without_glove'
            model_main, question_tokenizer_main, query_tokenizer_main, max_question_len_main, _ = load_model(model_dir_main)
        
            # Predict SQL query using the main model
            sql_query_main = predict_query(model_main, question_tokenizer_main, query_tokenizer_main, max_question_len_main, nlp_input_text)
            
            # Check for the appropriate model based on output generated by the main model
            match_select = re.search(r'\bselect\b', sql_query_main, re.IGNORECASE)
            match_order = re.search(r'\border by\b', sql_query_main, re.IGNORECASE)
            match_join = re.search(r'\bjoin\b', sql_query_main, re.IGNORECASE)
            
            if match_join:
                model_dir = './LSTM_join_glove'
            elif match_order:
                model_dir = './LSTM_order_glove' 
            elif match_select:
                model_dir = './LSTM_select_glove'
            else:
                st.write("Can't load model, check again.")
                return
            
            # Load the appropriate model
            model, question_tokenizer, query_tokenizer, max_question_len, _ = load_model(model_dir)
            st.write(f"using model{model_dir}")
            
            # Predict with the specific model and display result
            sql_query = predict_query(model, question_tokenizer, query_tokenizer, max_question_len, nlp_input_text)
            st.subheader("Generated SQL Query: ")
            st.code(sql_query)
            st.session_state.sql_query = sql_query
            
            # Execute query
            query_result = execute_query(sql_query)

            # Display results
            if query_result:
                st.write("Query Results:")
                for row in query_result:
                    st.write(row)
            else:
                st.write("No results found.")

            # Update session state to indicate query execution
            st.session_state.query_executed = True
            
    if st.session_state.query_executed:
        regenerate = st.button('Regenerate')
        
        if regenerate:
            question = nlp_input_text
            
            # Get the Gemini-generated SQL response
            response = get_gemini_response(question, prompt)
            st.subheader("Generated SQL Query: ")
            st.code(response)

            # Execute the regenerated query
            query_result = execute_query(response)

            # Display regenerated query results
            if query_result:
                st.subheader("Query Results: ")
                for row in query_result:
                    st.write(row)
            else:
                st.write("No results found.")

genai.configure(api_key="AIzaSyA3MGmDJg1QcXZ6lwb6tRtp16kPsSCVkoM")

if __name__ == "__main__":
    main()