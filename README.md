# NLP-to-SQL-query-bot
This project leverages LSTM models to convert natural language inputs, provided either as text or speech, into SQL queries. It allows users to interactively generate SQL commands based on conversational prompts, making database querying more accessible to those without SQL knowledge.

<br>

# Features
Convert natural language questions (e.g., “Show all the bikes from Honda”) to SQL queries.

Execute SQL queries on a target database.

Display results in a user-friendly format.

Support for common SQL operations (SELECT, WHERE, JOIN, GROUP BY, etc.).

Easy integration with frontend (Streamlit, Flask, etc.).

Optional: Voice input for questions.

<br>

# Key Architecture

A main LSTM model is trained to:

Classify the type of SQL query (e.g., ORDER, SELECT, JOIN) Generate a temporary draft query

- Three specialized sub-models:

  - ORDER model → Refines and generates ORDER queries

  - SELECT model → Refines and generates SELECT queries

  - JOIN model → Refines and generates JOIN queries

## How It Works
- The main model takes the user’s natural language question.

- It predicts the SQL query type and generates a draft query.

- Based on the predicted type, the draft is passed to the corresponding specialized sub-model.

- The sub-model produces the final, executable SQL query.

- The app executes the query on the database and displays the results.

- This modular approach improves query accuracy and makes the system more robust across different SQL operations.


<br>

# Project Structure

<pre> nlp-to-sql/
├── database/
│   ├── train_join.csv               # Questions containing only JOIN queries
│   ├── train_order.csv              # Questions containing only ORDER queries
│   ├── train_select.csv             # Questions containing only SELECT queries
│   └── train_new.csv                # Combined dataset with questions from simple to complex queries
├── app_regen_voice.py               # Main app (Streamlit) with regenerate and voice input features
├── train_sub_model.py               # Training script for sub-models (e.g., SELECT, JOIN, ORDER)
├── LSTM_train_main_model-glove.py   # Training script for main LSTM model (identifies query type and drafts initial query)
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation  </pre>

<br>

# Download Links
### Note - Models sizes are large so i have to put it on drive
### You can download the pretrained models along with their tokenizers, max lengths etc

- Link - https://drive.google.com/drive/folders/1XTAOWuPM0Q58GXhZENsw2clNePydEA7D?usp=drive_link

- just download the folder and put it in the project folder, no need to rename, everything would work out of the box.
- and ofcouse you have the choice to train your own model according to your need/choice.











