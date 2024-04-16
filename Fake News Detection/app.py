import streamlit as st
import pickle
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load pre-trained vectorizer and model
vector_form = pickle.load(open('vector2.pkl', 'rb'))
load_model = pickle.load(open('model2.pkl', 'rb'))

# Initialize Porter Stemmer
port_stem = PorterStemmer()

# Function to preprocess text
def preprocessing(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

# Function to extract news content from URL
def extract_news_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text content from the webpage
        news_content = ' '.join([p.get_text() for p in soup.find_all('p')])
        return news_content
    except Exception as e:
        print("Error:", e)
        return None

# Function to classify news as real or fake
def classify_news(news):
    news = preprocessing(news)
    input_data = [news]
    vectorized_data = vector_form.transform(input_data)
    prediction = load_model.predict(vectorized_data)
    return prediction[0]

# Function to collect user feedback and update the dataset
def collect_feedback(news, label):
    with open('dataset.pkl', 'ab') as file:
        pickle.dump((news, label), file)

# Function to retrain the model with updated dataset
def retrain_model():
    dataset = pickle.load(open('dataset.pkl', 'rb'))
    news_data = []
    labels = []
    for data in dataset:
        if isinstance(data, tuple):
            news, label = data
            news_data.append(preprocessing(news))
            labels.append(label)
        else:
            # Handle case where only one value is present in the tuple
            news_data.append(preprocessing(data))
            labels.append(None)  # Assigning None as a placeholder label
    vector_data = vector_form.transform(news_data)
    
    # Filter out entries with None labels
    filtered_indices = [i for i, label in enumerate(labels) if label is not None]
    filtered_vector_data = vector_data[filtered_indices]
    filtered_labels = [labels[i] for i in filtered_indices]

    print("Filtered vector data shape:", filtered_vector_data.shape)
    print("Filtered labels length:", len(filtered_labels))
    
    if len(filtered_labels) == 0:
        print("No data available for training.")
        return
    
    model = LogisticRegression()
    model.fit(filtered_vector_data, filtered_labels)
    # Save the updated model
    pickle.dump(model, open('model2.pkl', 'wb'))


if __name__ == '__main__':
    st.title('Fake News Classification App')

    # Input box for URL
    url = st.text_input("Enter the URL of the news article:")

    # Input box for news content
    news_content_input = st.text_area("Enter the news content:", height=200)

    # Button to trigger news classification
    predict_button = st.button("Classify News")

    if predict_button:
        if not url and not news_content_input:
            st.error("Please enter a URL or news content.")
        else:
            if url:
                # Extract news content from URL
                news_content = extract_news_from_url(url)
                if news_content:
                    # Display the extracted news content
                    st.subheader("Extracted News Content:")
                    st.write(news_content)
            else:
                news_content = news_content_input

            # Classify the news
            prediction_class = classify_news(news_content)
            if prediction_class == "REAL":
                st.success('The news is reliable.')
            elif prediction_class == "FAKE":
                st.warning('The news is unreliable.')

            # Collect user feedback
            feedback = st.radio("Is this news fake or real?", ('Fake', 'Real'))
            if feedback == 'Fake':
                collect_feedback(news_content, 'FAKE')
            elif feedback == 'Real':
                collect_feedback(news_content, 'REAL')
            # Retrain the model with updated data
            retrain_model()
            st.success("Model trained successfully with updated data.")

    # Section for adding new data to train the model
    st.header("Add New Data to Train the Model")
    new_url = st.text_input("Enter the URL of the news article to train the model:")
    new_text = st.text_area("Enter the news content to train the model:", height=200)
    if st.button("Submit as Fake"):
        if new_url:
            new_news_content = extract_news_from_url(new_url)
        else:
            new_news_content = new_text
        collect_feedback(new_news_content, 'FAKE')
        st.success("Data submitted successfully as fake.")
        retrain_model()
        st.success("Model trained successfully with updated data.")
    if st.button("Submit as Real"):
        if new_url:
            new_news_content = extract_news_from_url(new_url)
        else:
            new_news_content = new_text
        collect_feedback(new_news_content, 'REAL')
        st.success("Data submitted successfully as real.")
        retrain_model()
        st.success("Model trained successfully with updated data.")
