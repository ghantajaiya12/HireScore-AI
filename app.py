import streamlit as st
from utils.file_parser import extract_text
from utils.keyword_matcher import extract_keywords_and_match
from utils.llm_suggester import get_resume_suggestions
import matplotlib.pyplot as plt

st.set_page_config(page_title="HireScore AI", layout="wide")
st.title("HireScore AI")
st.caption("Smart resume matching powered by NLP and Generative AI")

col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area("Paste Job Description Here", height=200)

with col2:
    resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze") and jd_text and resume_file:
    resume_text = extract_text(resume_file, resume_file.name.split(".")[-1])


    score, missing = extract_keywords_and_match(jd_text, resume_text)
    suggestions = get_resume_suggestions(jd_text, resume_text)

    st.subheader(f"Match Score: {score}%")

    fig, ax = plt.subplots()
    ax.bar(["Match", "Missing"], [score, 100 - score], color=["green", "red"])
    st.pyplot(fig)

    st.markdown(f"### Missing Keywords:\n{', '.join(missing)}")
    st.subheader("AI Suggestions to Improve Resume")
    st.write(suggestions)
else:
    st.info("Please upload both files and click Analyze.")

st.markdown("---")
st.caption("Built with Python, Streamlit & HuggingFace by You!")
