import streamlit as st
import pickle
import re
import nltk
import requests
from groq import Groq
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ⚠️ Put your API keys here
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

# Setup Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Fetch related real-time news
def fetch_related_news(query):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=3&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return articles
    except:
        return []

# Ask Gemini to analyze the news
def analyze_with_gemini(news_text, ml_prediction, confidence):
    prompt = f"""
    You are a professional fact-checker specializing in Indian and global news.
    
    Analyze this news: "{news_text}"
    
    Our ML model says it is: {"REAL" if ml_prediction == 1 else "FAKE"} with {confidence:.2%} confidence.
    
    Please provide:
    1. Your own verdict (Real/Fake/Uncertain)
    2. Key reasons why (2-3 points)
    3. What to watch out for in this type of news
    4. Advice for the reader
    
    Keep it concise and simple. Use plain English.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake News Detector")
st.write("Powered by ML + Google Gemini AI + Real-time News")
st.divider()

user_input = st.text_area("📰 Paste your news headline or article here:", height=200)

if st.button("🔍 Analyze News", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):

            # Step 1: ML Model prediction
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized)[0]
            ml_confidence = confidence[1] if prediction == 1 else confidence[0]

            st.subheader("📊 ML Model Result")
            if prediction == 1:
                st.success(f"✅ REAL NEWS — {ml_confidence:.2%} confident")
            else:
                st.error(f"❌ FAKE NEWS — {ml_confidence:.2%} confident")

            st.divider()

            # Step 2: Gemini AI Analysis
            st.subheader("🤖 AI Analysis (Gemini)")
            gemini_result = analyze_with_gemini(user_input, prediction, ml_confidence)
            st.write(gemini_result)

            st.divider()

            # Step 3: Related real-time news
            st.subheader("🌐 Related Real-Time News")
            articles = fetch_related_news(user_input[:50])
            if articles:
                for article in articles:
                    st.markdown(f"**{article['title']}**")
                    st.caption(f"{article['source']['name']} — {article['publishedAt'][:10]}")
                    st.write(article['description'])
                    st.markdown(f"[Read more]({article['url']})")
                    st.divider()
            else:
                st.info("No related news found at this time.")