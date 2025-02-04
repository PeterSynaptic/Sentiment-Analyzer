import streamlit as st
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_sentiment(text):
    """Analyzes text sentiment using Gemini API with enhanced error handling."""
    try:
        # Configure API (use secrets management for production)
        api_key = st.secrets.get("API_KEY") or st.text_input("Enter your Gemini API Key:", type="password")
        if not api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()

        genai.configure(api_key=api_key) 

        
        # Model configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )

        # Start chat with formatted history
        chat_session = model.start_chat(history=[
            {"role": "user", "parts": ["""
                Analyze text sentiment and return JSON with 'sentiment' (positive/negative/neutral) 
                and 'reason' (concise explanation). Examples:
                {
                    "text": "Great product! Works perfectly.",
                    "sentiment": "positive",
                    "reason": "Positive adjectives and enthusiastic tone"
                }
            """]},
            {"role": "model", "parts": ["Understood. I'll return JSON responses with sentiment analysis."]}
        ])

        # Get and parse response
        response = chat_session.send_message(text)
        return json.loads(response.text)

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# Streamlit UI Configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# App Header
st.title("📊 Sentiment Analysis Tool")
st.markdown("Analyze text sentiment using advanced NLP")

# Input Section
with st.container():
    user_input = st.text_area(
        "Enter your text:",
        height=150,
        placeholder="Paste your review, comment, or text here..."
    )

# Analysis Section
if st.button("Analyze Sentiment", type="primary", use_container_width=True, **{'secondary': True}):
        if user_input:
            with st.spinner("🔍 Analyzing sentiment..."):
                result = analyze_sentiment(user_input)
            
        if result:
            # Check for list response format
            if isinstance(result, list):
                result = result[0]
                
            with st.container(border=True):
                st.subheader("Analysis Results")
                
                # Sentiment badge
                sentiment_color = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "blue"
                }.get(result['sentiment'].lower(), "gray")
                
                st.markdown(f"""
                    **Sentiment**:  
                    <span style="
                        background-color: {sentiment_color}20;
                        color: {sentiment_color};
                        padding: 0.2rem 0.5rem;
                        border-radius: 0.25rem;
                        font-weight: 500;
                    ">{result['sentiment'].capitalize()}</span>
                """, unsafe_allow_html=True)
                
                # Reason with scrollable container
                st.markdown("**Analysis:**")
                st.markdown(f"""
                    <div style="
                        max-height: 200px;
                        overflow-y: auto;
                        padding: 0.5rem;
                        background-color: #f8f9fa;
                        border-radius: 0.5rem;
                    ">
                    {result['reason']}
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze")


