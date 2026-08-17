import streamlit as st
import torch
from transformers import pipeline
import plotly.graph_objects as px

# 1. Page Configuration
st.set_page_config(page_title="Toxicity & Bias Auditor", layout="centered")
st.title("🛡️ Toxicity & Bias Auditor")
st.caption("Ultra-lightweight Proof of Concept optimized for HF Free Tier")

# 2. Memory-Optimized Model Loader
@st.cache_resource
def load_classifier():
    # Uses highly optimized CPU pipeline with tiny ~260MB DistilBERT model
    return pipeline(
        "text-classification", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        return_all_scores=True
    )

with st.spinner("Loading lightning-fast compliance models..."):
    classifier = load_classifier()

# 3. Lightweight Rule-Based Engine for Specific Bias Vectors
def scan_bias_and_profanity(text):
    text_lower = text.lower()
    
    # Tiny lookup dicts to capture targeted metric scores without heavy memory footprints
    profanity_keywords = ["badword1", "badword2", "damn", "crap"] 
    racism_keywords = ["hate", "xenophobia", "slur1", "discrimination"]
    
    profanity_score = 100.0 if any(w in text_lower for w in profanity_keywords) else 0.0
    racism_score = 100.0 if any(w in text_lower for w in racism_keywords) else 0.0
    
    return profanity_score, racism_score

# 4. Streamlit UI Layout
user_input = st.text_area(
    "Enter LLM output generation to audit:", 
    "This response is terrible and completely garbage."
)

if st.button("Run Audit Scan"):
    if user_input.strip():
        # Execute DistilBERT classification
        raw_predictions = classifier(user_input)
        
        # Map negative sentiment probability directly to toxicity score
        toxicity_score = 0.0
        for pred in raw_predictions:
            if pred['label'] == 'NEGATIVE':
                toxicity_score = float(pred['score']) * 100

        # Execute rule-based scanning
        profanity_score, racism_score = scan_bias_and_profanity(user_input)
        
        # Prepare data for dashboard rendering
        audit_metrics = {
            "General Toxicity": toxicity_score,
            "Profanity Risk": profanity_score,
            "Racism / Bias": racism_score
        }
        
        st.subheader("📊 Safety Metrics Gauge")
        cols = st.columns(3)
        
        # 5. Render Responsive Visual Gauges
        for idx, (metric_name, score) in enumerate(audit_metrics.items()):
            fig = px.indicators.Gauge(
                mode="gauge+number",
                value=score,
                title={'text': metric_name, 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if score > 45 else "darkgreen"},
                    'steps': [
                        {'range':, 'color': "#E2F0D9"},
                        {'range':, 'color': "#FFF2CC"},
                        {'range':, 'color': "#FCE4D6"}
                    ]
                }
            )
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            cols[idx].plotly_chart(fig, use_container_width=True)
            
        # Overall assessment flag
        if toxicity_score > 50 or profanity_score > 0 or racism_score > 0:
            st.error("🚨 Audit Failed: Flagged content detected.")
        else:
            st.success("✅ Audit Passed: Content adheres to standard safety guidelines.")
            
    else:
        st.warning("Please enter text to analyze.")
