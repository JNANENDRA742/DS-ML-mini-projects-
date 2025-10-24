🚀 Features

Analyze sentiment of reviews, tweets, comments, or any text.
Supports multi-language input via translation.
Handles emoji inputs and converts them to text for analysis.
Displays overall sentiment and the most influential sentence.
Shows probability distribution of sentiment scores using bar charts.
Interactive and user-friendly Streamlit interface with custom dark theme.
Users can rate the app with a feedback system.

🛠️ Technologies Used

Python 3.x
Streamlit for UI
Transformers (RoBERTa) from Hugging Face
VADER for rule-based sentiment analysis
NLTK for sentence tokenization
Deep Translator for translation
Emoji module for emoji processing
PyTorch for deep learning model inference

🔧 How It Works

User enters text in any language (including emojis).
Emojis are converted to text, and text is translated to English if necessary.
Text is cleaned by removing URLs, mentions, hashtags, and extra spaces.
Sentiment is analyzed using RoBERTa.
App identifies the sentence contributing most to the overall sentiment.

Results are displayed:
Overall Sentiment (Positive, Negative, Neutral)
Most Influential Sentence
Probability Distribution chart
Raw sentiment scores
