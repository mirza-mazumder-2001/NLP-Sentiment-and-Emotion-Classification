import streamlit as st
import pickle
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download nltk resources

nltk.download('punkt_tab')
nltk.download('stopwords')

# page configuration

st.set_page_config(
    page_title="NLP AI Classifier",
    page_icon="🤖",
    layout="wide"
)

# custom css styling

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #4f46e5;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #475569;
    margin-bottom: 40px;
}

.stTextArea textarea {
    border-radius: 15px;
    font-size: 18px;
    border: 2px solid #6366f1;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# title section

st.markdown(
    '<div class="title">🤖 NLP AI Text Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Sentiment Analysis & Emotion Classification using Machine Learning and NLP</div>',
    unsafe_allow_html=True
)

# sidebar

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2103/2103832.png",
    width=120
)

st.sidebar.title("⚙️ Settings")

option = st.sidebar.radio(
    "Choose Prediction Type",
    [
        "Sentiment Analysis",
        "Emotion Classification"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    This application uses:
    
    ✔ NLP Preprocessing  
    ✔ TF-IDF Vectorization  
    ✔ Logistic Regression  
    ✔ Machine Learning Models  
    """
)

# initialize NLP tools

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# preprocessing function

def preprocess_text(text):

    text = text.lower()

    words = word_tokenize(text)

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(words)

# text input

user_input = st.text_area(
    "📝 Enter Your Text",
    height=200,
    placeholder="Type something here..."
)

# predict button

predict_button = st.button("🚀 Predict", use_container_width=True)

# prediction logic

if predict_button:

    if user_input.strip() == "":

        st.warning("⚠ Please enter some text.")

    else:

        cleaned_text = preprocess_text(user_input)

        # sentiment analysis

        if option == "Sentiment Analysis":

            with open('sentiment_model.pkl', 'rb') as file:
                sentiment_model = pickle.load(file)

            with open('sentiment_vectorizer.pkl', 'rb') as file:
                sentiment_vectorizer = pickle.load(file)

            vectorized_text = sentiment_vectorizer.transform([cleaned_text])

            prediction = sentiment_model.predict(vectorized_text)[0]

            # positive

            if prediction.lower() == "positive":

                st.balloons()

                st.markdown(
                    """
                    <div class="result-box" style="background-color:#dcfce7; color:#166534;">
                    😊 Positive Sentiment
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # negative

            elif prediction.lower() == "negative":

                st.markdown(
                    """
                    <div class="result-box" style="background-color:#fee2e2; color:#991b1b;">
                    😞 Negative Sentiment
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # neutral

            else:

                st.markdown(
                    """
                    <div class="result-box" style="background-color:#e0f2fe; color:#075985;">
                    😐 Neutral Sentiment
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # emotion classification

        else:

            with open('emotion_model.pkl', 'rb') as file:
                emotion_model = pickle.load(file)

            with open('emotion_vectorizer.pkl', 'rb') as file:
                emotion_vectorizer = pickle.load(file)

            vectorized_text = emotion_vectorizer.transform([cleaned_text])

            prediction = emotion_model.predict(vectorized_text)[0]

            emotion_styles = {

                "joy": ("😊 Joy", "#fef9c3", "#854d0e"),

                "sadness": ("😢 Sadness", "#dbeafe", "#1e3a8a"),

                "anger": ("😠 Anger", "#fee2e2", "#991b1b"),

                "fear": ("😨 Fear", "#ede9fe", "#5b21b6"),

                "love": ("❤️ Love", "#ffe4e6", "#be123c"),

                "surprise": ("😲 Surprise", "#ecfccb", "#3f6212")
            }

            label, bg_color, text_color = emotion_styles.get(
                prediction,
                ("🙂 Emotion", "#f3f4f6", "#111827")
            )

            st.markdown(
                f"""
                <div class="result-box"
                style="background-color:{bg_color}; color:{text_color};">
                {label}
                </div>
                """,
                unsafe_allow_html=True
            )

# footer

st.markdown("---")

st.markdown(
    """
    <div class="footer">
    Developed by Mirza Hussain Mazumder, using Streamlit, NLP, and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)