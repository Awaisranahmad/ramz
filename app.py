import streamlit as st
from groq import Groq, RateLimitError
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import xml.etree.ElementTree as ET
import re
import time

# ---------------- API ----------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title="AI File Detector", page_icon="🛡️", layout="wide")

# ---------------- FILE READER ----------------

def read_file(file):

    name = file.name.lower()

    try:

        if name.endswith(".pdf"):
            pdf = PdfReader(file)
            return " ".join(p.extract_text() or "" for p in pdf.pages)

        elif name.endswith(".docx"):
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)

        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            return df.to_string()

        elif name.endswith(".xlsx"):
            df = pd.read_excel(file)
            return df.to_string()

        elif name.endswith(".json"):
            data = json.load(file)
            return json.dumps(data, indent=2)

        elif name.endswith(".xml"):
            tree = ET.parse(file)
            root = tree.getroot()
            return ET.tostring(root, encoding="unicode")

        else:
            return file.read().decode(errors="ignore")

    except Exception as e:
        return str(e)


# ---------------- STYLE ANALYSIS ----------------

def style_analysis(text):

    sentences = re.split(r'[.!?]', text)
    words = text.split()

    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

    avg_sentence = np.mean(sentence_lengths) if sentence_lengths else 0
    variance = np.var(sentence_lengths) if sentence_lengths else 0

    vocab = len(set(words))
    total = len(words)

    diversity = vocab / total if total else 0

    return avg_sentence, variance, diversity


# ---------------- GROQ SAFE REQUEST ----------------

def ask_ai(prompt):

    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            return response.choices[0].message.content

        except RateLimitError:
            time.sleep(5)

    return "AI_PERCENT: 50\nENGINE: Unknown\nCONFIDENCE: Low\nREPORT: Rate limit reached."


# ---------------- UI ----------------

st.title("🛡️ Universal AI Content Detector")

col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area("Paste content")

with col2:
    file = st.file_uploader("Upload any file")

content = ""

if file:
    content = read_file(file)
else:
    content = text_input

analyze = st.button("Run AI Scan")

# ---------------- ANALYSIS ----------------

if analyze and content:

    content = content[:2000]

    avg, var, div = style_analysis(content)

    prompt = f"""
You are an AI forensic investigator.

Analyze if this text was written by AI.

Return EXACT format:

AI_PERCENT: number
ENGINE: name
CONFIDENCE: High Medium or Low
REPORT: explanation

TEXT:
{content}
"""

    result = ask_ai(prompt)

    ai_match = re.search(r"AI_PERCENT:\s*(\d+)", result)
    ai = int(ai_match.group(1)) if ai_match else 50
    human = 100 - ai

    engine_match = re.search(r"ENGINE:\s*(.*)", result)
    engine = engine_match.group(1) if engine_match else "Unknown"

    conf_match = re.search(r"CONFIDENCE:\s*(.*)", result)
    conf = conf_match.group(1) if conf_match else "Medium"

    report_match = re.search(r"REPORT:\s*(.*)", result, re.S)
    report = report_match.group(1) if report_match else "Analysis complete."

    # ---------------- CHART 1 ----------------

    fig = go.Figure(data=[go.Pie(
        labels=["AI","Human"],
        values=[ai,human],
        hole=.6
    )])

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- CHART 2 ----------------

    fig2 = go.Figure()

    fig2.add_bar(
        x=["Sentence Variance","Vocabulary Diversity","Avg Sentence Length"],
        y=[var, div*100, avg]
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- STATS ----------------

    st.subheader("Text Stats")

    c1, c2, c3 = st.columns(3)

    c1.metric("Words", len(content.split()))
    c2.metric("Unique Words", len(set(content.split())))
    c3.metric("Characters", len(content))

    # ---------------- ENGINE ----------------

    st.subheader("Detected Engine")

    st.info(engine)

    st.write("Confidence:", conf)

    # ---------------- REPORT ----------------

    st.subheader("Forensic Report")

    st.write(report)

elif analyze:
    st.warning("Please upload file or paste text.")
