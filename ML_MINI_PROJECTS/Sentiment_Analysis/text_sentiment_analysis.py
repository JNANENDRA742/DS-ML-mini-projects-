# -------------------- Import Required Libraries --------------------
import os
import re
import time
import nltk
import torch
import emoji
from nltk import sent_tokenize
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from deep_translator import GoogleTranslator

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
# -------------------- Setup Model Directory --------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_PATH = "./saved_roberta_model"

# -------------------- Setup Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Text Cleaning --------------------
def clean_text(text):
    """Remove URLs, hashtags, mentions, and extra spaces."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.strip()
    return text

# -------------------- Load Model --------------------
@st.cache_resource
def load_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    return tokenizer, model

# -------------------- Sentiment Analysis --------------------
def analyze_sentiment(text, tokenizer, model):
    """Analyze sentiment of given text (supports emojis)."""
    text = clean_text(text)
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model(**encoded_text)

    scores = softmax(output.logits[0].detach().cpu().numpy())
    labels = ['Negative', 'Neutral', 'Positive']
    results = {label: round(float(score), 4) for label, score in zip(labels, scores)}
    sentiment = labels[scores.argmax()]
    return sentiment, results


# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# -------------------- Custom Dark Theme CSS --------------------
st.markdown("""
<style>
* {
    font-family: 'Segoe UI', sans-serif;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0d0d0d;
    color: #e5e5e5;
}

/* Header */
[data-testid="stHeader"] {
    background: none;
}

/* Title */
h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.3em;
}

/* Paragraphs, Labels, Spans */
p, label, span {
    color: #d1d5db !important;
    font-size: 16px;
}

/* Text Area */
textarea {
    background-color: #1a1a1a !important;
    color: #f5f5f5 !important;
    border: 1.5px solid #2f2f2f !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    padding: 10px !important;
}

/* Analyze Button */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    transform: scale(1.02);
}

/* Sentiment Styles */
.positive {
    color: #00e676;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    text-shadow: 0 0 15px #00e676;
}
.negative {
    color: #ff1744;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    text-shadow: 0 0 15px #ff1744;
}
.neutral {
    color: #ffea00;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    text-shadow: 0 0 15px #ffea00;
}
/* Chart heading */
.gradient-heading {
    font-size: 22px;
    color: #e5e5e5;
    text-align: center;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* JSON and chart containers */
.stJson {
    background-color: #1a1a1a !important;
    border-radius: 8px !important;
    border: 1px solid #333333 !important;
    padding: 10px !important;
}

hr {
    border: 1px solid #2f2f2f;
    margin-top: 1.5em;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



# -------------------- Load Model --------------------
tokenizer, model = load_roberta_model()
# ---------------------checking if the input contains only emojis or not---------------------------
def is_only_emoji(text):
    no_emoji_text = emoji.replace_emoji(text,replace="").strip()
    return len(no_emoji_text)==0

# -------------------- Title --------------------
st.title("💫 Welcome to Sentiment Analyzer App")
st.write("""This app analyzes the **sentiment** of your text Instantly 😊""")

user_input = st.text_area( "✏️ Enter your Review or Comment to Analyze:", placeholder="🔍Type something like — 'I love this product!'", height=150)

# ------------converting emoji into the related text------------------
if is_only_emoji(user_input):
    print("Input contains emojis")
    result = emoji.demojize(user_input,language='en')
    print(result)
    user_input = re.sub(r':','',result)
    
# -------------------- Translation Safety -------------------- 
translated_text = None 
if user_input and user_input.strip():
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(user_input)
        st.write(translated_text)
    except Exception as e:
        st.warning("⚠️ Please check your internet connection.But Don’t worry — we analyzed your text directly.")
        translated_text = user_input

else:
    translated_text = ""

# -------------------- Analyze Button -------------------- 
if st.button("🔍 Analyze Sentiment"): 
    st.toast("⏳ Analyzing sentiment... please wait 😊")
    time.sleep(0.1)
    
    if not translated_text or translated_text.strip() == "": 
        st.warning("⚠️ Please enter some text before analyzing.") 
    else: 
        with st.spinner("⏳ Analyzing sentiment... please wait 😊"):
            time.sleep(0.5)
            sentiment, scores = analyze_sentiment(translated_text, tokenizer, model) 

            # ---------finding the sentence which influenced the sentiment------------------
            sentences = sent_tokenize(translated_text)
            sentiment_scores = []

            for s in sentences:
                inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=512).to(device)
                outputs = model(**inputs)
                sc = softmax(outputs.logits[0].detach().cpu().numpy())
                sentiment_scores.append((s, sc))

            # Find sentence with maximum sentiment change
            max_change_sentence, max_change_scores = max(
                sentiment_scores, key=lambda x: abs(x[1][2] - x[1][0])
            )

            # Determine sentiment of that sentence
            labels = ["Negative", "Neutral", "Positive"]
            sentence_sentiment = labels[max_change_scores.argmax()]

            # Proper color mapping
            color_class1 = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}[sentiment]
            color_class2 = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}[sentence_sentiment]

            # Display results
            st.markdown(f"<p class='{color_class1}'>💬 Overall Sentiment: {sentiment}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='{color_class2}'>💡 Most Influential Sentence: “{max_change_sentence}”</p>", unsafe_allow_html=True)

            # Bar chart heading
            st.markdown("<h3 class='gradient-heading'>📊 Sentiment Probability Distribution</h3>", unsafe_allow_html=True)
            st.bar_chart(scores)

            # Display raw scores
            st.write("📊 Sentiment Scores:")
            st.json(scores)


# -----------------------Footer--------------------------- 
st.markdown("---") 

    # st.info("")

# ----------------------------------
# Feedback Page
st.title("⭐ Rate This App")
st.write("How was your experience using the Sentiment Analyzer?")

    # Feedback widget
select = st.feedback("stars")

stars_count = ["one", "two", "three", "four", "five"]

if select is not None:
    index = int(select) 
    st.markdown(f"You have selected **{stars_count[index]} stars** ⭐")

        # Custom thank-you messages
    messages = [
            "😞 Sorry to hear that. We’ll try to improve!",
            "🙂 Thanks! We’ll make it better.",
            "😊 Glad you liked it!",
            "😄 Awesome! Thanks for your feedback!",
            "🤩 You made our day! Thank you so much!"
        ]
    st.info(messages[index])


