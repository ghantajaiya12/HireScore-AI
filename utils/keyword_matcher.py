from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_keywords_and_match(jd_text, resume_text):
    vectorizer = CountVectorizer().fit_transform([jd_text, resume_text])
    vectors = vectorizer.toarray()
    score = round(cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100, 2)

    jd_keywords = set(jd_text.lower().split())
    resume_keywords = set(resume_text.lower().split())
    missing = list(jd_keywords - resume_keywords)
    return score, missing[:15]
