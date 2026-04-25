import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Gemini
import google.generativeai as genai

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="FairAI Pro", layout="wide")
pio.templates.default = "plotly_dark"

# =========================
# UI STYLE (RESTORED 🔥)
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🤖 FairAI Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Bias Detection • Ethical AI • AI Insights</p>", unsafe_allow_html=True)

# =========================
# GEMINI SETUP
# =========================
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key="AIzaSyCn8xdjjpEfq9RponBsjfckOEbDUi7DoPo")
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
else:
    model_gemini = None

# =========================
# LOAD MODEL
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    st.sidebar.success("Dataset Loaded ✅")

    target = st.sidebar.selectbox("🎯 Target", df.columns)
    sensitive = st.sidebar.selectbox("⚠️ Sensitive Attribute", df.columns)

    if target == sensitive:
        st.warning("Target and Sensitive must be different")
        st.stop()

    # =========================
    # TARGET FIX
    # =========================
    if df[target].dtype == "object":
        df[target] = df[target].map({
            'Y':1,'N':0,
            'Yes':1,'No':0,
            'Approved':1,'Rejected':0
        })

    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target])

    # =========================
    # METRICS
    # =========================
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Groups", df[sensitive].nunique())

    st.dataframe(df.head())

    # =========================
    # BUTTONS
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    analyze_btn = c1.button("🔍 Analyze")
    predict_btn = c2.button("🤖 Predict")
    mitigate_btn = c3.button("🛠 Mitigate")
    report_btn = c4.button("📑 Report")

    # =========================
    # ANALYSIS
    # =========================
    if analyze_btn:

        group_rates = df.groupby(sensitive)[target].mean()
        di = group_rates.min() / group_rates.max()
        diff = group_rates.max() - group_rates.min()

        st.metric("Disparate Impact", round(di,3))
        st.metric("Difference", round(diff,3))

        fig = px.bar(x=group_rates.index, y=group_rates.values)
        st.plotly_chart(fig)

        if model_gemini:
            try:
                with st.spinner("🤖 Generating AI insights..."):
                    prompt = f"""
                    Explain fairness results simply.
                    DI: {di}
                    Difference: {diff}
                    Data: {group_rates.to_dict()}
                    """
                    response = model_gemini.generate_content(prompt)
                st.write(response.text)

            except Exception as e:
                st.error(f"Gemini error: {e}")

    # =========================
    # PREDICT
    # =========================
    if predict_btn:
        df_model = pd.get_dummies(df, drop_first=True)
        df_model = df_model.reindex(columns=model_columns, fill_value=0)

        df["Prediction"] = model.predict(df_model)
        st.dataframe(df.head())

    # =========================
    # MITIGATION
    # =========================
    if mitigate_btn:
        st.dataframe(df.drop(columns=[sensitive]))

    # =========================
    # REPORT (FIXED 🔥)
    # =========================
    if report_btn:

        group_rates = df.groupby(sensitive)[target].mean()
        di = group_rates.min() / group_rates.max()

        # Create chart
        chart_path = "chart.png"
        plt.figure(figsize=(5,3))
        group_rates.plot(kind='bar')
        plt.title("Fairness Across Groups")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        # Gemini summary
        ai_text = "AI summary not available."
        if model_gemini:
            try:
                prompt = f"Write fairness report summary. DI={di}, Data={group_rates.to_dict()}"
                response = model_gemini.generate_content(prompt)
                ai_text = response.text
            except Exception as e:
                ai_text = f"Gemini failed: {e}"

        # PDF
        pdf_path = "FairAI_Report.pdf"
        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        content = []

        # PAGE 1
        content.append(Paragraph("FAIRAI PRO REPORT", styles['Title']))
        content.append(Spacer(1, 20))

        content.append(Paragraph(f"Generated: {datetime.now()}", styles['Normal']))
        content.append(Spacer(1, 20))

        content.append(Paragraph("1. Dataset Summary", styles['Heading2']))
        content.append(Paragraph(f"Records: {df.shape[0]}", styles['Normal']))
        content.append(Spacer(1, 20))

        content.append(Paragraph("2. Fairness Metrics", styles['Heading2']))
        content.append(Paragraph(f"Disparate Impact: {round(di,3)}", styles['Normal']))
        content.append(Spacer(1, 20))

        # PAGE 2
        content.append(PageBreak())

        content.append(Paragraph("3. Visualization", styles['Heading2']))
        content.append(Spacer(1, 20))

        img = Image(chart_path)
        img.drawHeight = 220
        img.drawWidth = 400
        content.append(img)

        content.append(Spacer(1, 30))

        content.append(Paragraph("4. AI Insights", styles['Heading2']))
        content.append(Spacer(1, 10))
        content.append(Paragraph(ai_text, styles['Normal']))

        doc.build(content)

        st.success("✅ PDF Generated")

        with open(pdf_path, "rb") as f:
            st.download_button("⬇ Download PDF", f, "FairAI_Report.pdf")