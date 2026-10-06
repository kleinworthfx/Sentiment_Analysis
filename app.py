import re
from typing import List, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Sentiment Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@st.cache_resource(show_spinner=False)
def load_chat_model():
    return pipeline("text-generation", model="distilgpt2")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def classify_texts(texts: List[str]) -> pd.DataFrame:
    classifier = load_sentiment_model()
    cleaned = [clean_text(item) for item in texts if clean_text(item)]
    if not cleaned:
        return pd.DataFrame(columns=["text", "label", "score", "sentiment"])
    results = classifier(cleaned, truncation=True, max_length=512)
    rows = []
    for text, result in zip(cleaned, results):
        label = result["label"].upper()
        rows.append(
            {
                "text": text,
                "label": label,
                "score": round(float(result["score"]), 4),
                "sentiment": "Positive" if label == "POSITIVE" else "Negative",
            }
        )
    return pd.DataFrame(rows)


def make_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "The onboarding experience was smooth and genuinely helpful.",
                "Support solved my issue in minutes. Excellent service!",
                "The latest update is confusing and made the workflow slower.",
                "I love the clean design and fast performance.",
                "The delivery arrived late and the packaging was damaged.",
                "Everything worked as expected, nothing to complain about.",
                "The documentation needs more practical examples.",
                "A wonderful product that our whole team enjoys using.",
            ]
        }
    )


def render_overview() -> None:
    st.title("Sentiment Studio")
    st.caption("A practical AI workspace for understanding feedback and exploring language models.")

    with st.sidebar:
        st.header("Workspace")
        page = st.radio("Choose a view", ["Analyze", "Batch insights", "AI assistant"])
        st.divider()
        st.markdown("**Models**")
        st.caption("Sentiment: DistilBERT SST-2")
        st.caption("Assistant: DistilGPT-2")
        st.divider()
        st.info("Models download on first use and are cached for the session.")

    if page == "Analyze":
        render_analyze()
    elif page == "Batch insights":
        render_batch()
    else:
        render_assistant()


def render_analyze() -> None:
    st.subheader("Analyze a message")
    text = st.text_area(
        "Enter customer feedback, a review, or any short message",
        height=150,
        placeholder="Example: The product is intuitive and saved our team hours every week.",
    )
    if st.button("Analyze sentiment", type="primary", disabled=not text.strip()):
        with st.spinner("Running the sentiment model..."):
            result = classify_texts([text])
        row = result.iloc[0]
        color = "#16a34a" if row["sentiment"] == "Positive" else "#dc2626"
        st.markdown(
            f'<div style="padding:1.2rem;border-radius:0.8rem;background:{color}18;border:1px solid {color}55">'
            f'<h2 style="color:{color};margin:0">{row["sentiment"]}</h2>'
            f'<p style="margin:.35rem 0 0">Confidence: <b>{row["score"]:.1%}</b></p></div>',
            unsafe_allow_html=True,
        )
        st.progress(float(row["score"]), text=f"Model confidence: {row['score']:.1%}")


def render_batch() -> None:
    st.subheader("Batch insights")
    st.write("Paste one item per line or start with the included sample feedback.")
    if st.button("Load sample feedback"):
        st.session_state["batch_text"] = "\n".join(make_sample_data()["text"].tolist())
    batch_text = st.text_area(
        "Feedback items",
        value=st.session_state.get("batch_text", ""),
        height=220,
        placeholder="One feedback item per line...",
    )
    if st.button("Analyze batch", type="primary", disabled=not batch_text.strip()):
        items = [line for line in batch_text.splitlines() if line.strip()]
        with st.spinner(f"Analyzing {len(items)} items..."):
            result = classify_texts(items)
        st.session_state["batch_result"] = result

    result = st.session_state.get("batch_result")
    if result is None or result.empty:
        return
    positive = int((result["sentiment"] == "Positive").sum())
    negative = len(result) - positive
    avg_confidence = result["score"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total analyzed", len(result))
    c2.metric("Positive", positive, f"{positive / len(result):.0%} of total")
    c3.metric("Average confidence", f"{avg_confidence:.1%}")

    chart_data = result["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    fig = px.bar(
        chart_data,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map={"Positive": "#16a34a", "Negative": "#dc2626"},
        title="Sentiment distribution",
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button(
        "Download results as CSV",
        result.to_csv(index=False).encode("utf-8"),
        "sentiment_results.csv",
        "text/csv",
    )


def render_assistant() -> None:
    st.subheader("AI assistant")
    st.caption("Ask for help interpreting feedback or drafting a response. The lightweight model runs locally in the app.")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for message in st.session_state["chat"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    prompt = st.chat_input("Ask the assistant something...")
    if prompt:
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    generator = load_chat_model()
                    context = "You are a helpful feedback analyst. Give concise, practical answers.\nUser: " + prompt + "\nAssistant:"
                    output = generator(context, max_new_tokens=100, do_sample=True, temperature=0.7, pad_token_id=50256)
                    answer = output[0]["generated_text"].split("Assistant:")[-1].strip()
                    answer = answer or "I can help summarize feedback, identify themes, or draft a thoughtful response."
                except Exception:
                    answer = "I can help summarize feedback, identify themes, or draft a thoughtful response. Try asking about a sentiment result or customer theme."
                st.write(answer)
        st.session_state["chat"].append({"role": "assistant", "content": answer})


render_overview()
