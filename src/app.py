from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import streamlit as st
import zipfile
from collections import Counter
import fitz  # PyMuPDF
import os
import altair as alt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import unicodedata

# --- Extraction texte PDF (bytes) ---
def extract_text_from_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

# --- Parsing mots-clés pondérés ---
def parse_keywords(text):
    kw = {}
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":")
            try:
                kw[k.strip().lower()] = float(v.strip())
            except:
                kw[k.strip().lower()] = 1.0
        else:
            kw[part.strip().lower()] = 1.0
    return kw

# --- Scoring texte selon mots-clés pondérés ---
def score_text(text, keywords_weights):
    words = text.split()
    word_counts = Counter(words)
    score = 0.0
    for kw, weight in keywords_weights.items():
        score += word_counts.get(kw.lower(), 0) * weight
    return score

# --- Nettoyage du nom de fichier en nom propre ---
def clean_filename(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext.lower().startswith("cv_"):
        name_without_ext = name_without_ext[3:]
    name_parts = name_without_ext.replace("-", " ").replace("_", " ").split()
    name_parts = [part.capitalize() for part in name_parts]
    return " ".join(name_parts)

# --- Compte occurrences mots-clés dans un texte ---
def count_keywords_in_text(text, keywords):
    words = text.lower().split()
    counts = Counter()
    for kw in keywords.keys():
        counts[kw] = words.count(kw.lower())
    return counts

def clean_text(text):
    # Normalisation Unicode : transforme les lettres accentuées en lettre + accent
    text = unicodedata.normalize('NFD', text)
    # Supprime les accents
    text = ''.join([char for char in text if not unicodedata.combining(char)])
    # Lowercase
    text = text.lower()
    # Supprime la ponctuation et chiffres (ajuste si besoin)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Supprime les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text, top_k=10):
    cleaned_text = clean_text(text)
    french_stopwords = stopwords.words('french')
    
    vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=100)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    tfidf_dict = {term: score for term, score in zip(feature_names, scores)}
    
    # Top k par score brut
    top_terms = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

    max_score = top_terms[0][1] if top_terms else 1

    df = pd.DataFrame(top_terms, columns=["Mots_clés", "Score de suggestions (en %)"])
    df["Score de suggestions (en %)"] = (df["Score de suggestions (en %)"] / max_score) * 100
    df["Score de suggestions (en %)"] = df["Score de suggestions (en %)"].round(2)
    
    return df

# --- Interface Streamlit ---
st.title("🧠 Analyse intelligente de CV")
st.subheader("Votre assistant flexible pour la détection des talents")

st.write("""
Optimisez vos recrutements grâce à un outil simple, rapide et efficace :

📁 **1. Téléversez un fichier `.zip` contenant les CV au format PDF**  
📝 **2. Définissez vos critères de sélection via une liste de mots-clés pondérés**  
🔍 **3. Obtenez un score pour chaque CV basé sur la pertinence des mots-clés**  
✉️ **4. Contactez les candidats les plus prometteurs !**
""")

uploaded_zip = st.file_uploader("Téléversez le ZIP des CV PDF", type=["zip"])

job_description = st.text_area("Collez votre offre d'emploi ici (texte brut)", height=250)

if job_description:
    top_k = st.number_input("Nombre de mots-clés à extraire de l'offre d'emploi", min_value=1, max_value=50, value=10)

keywords_input = st.text_input("Mots-clés et poids (ex: python:3, docker:2, sql)")

top = st.number_input("Sélectionnez votre top des candidats à analyser", min_value=1, max_value=100, value=10)

if job_description:
    df_keywords = extract_keywords(job_description, top_k)
    st.markdown("### Mots-clés suggérés")
    st.dataframe(df_keywords, use_container_width=True, hide_index=True)

    # Prépare la chaîne à coller dans la zone mots-clés pondérés
    suggestion = ", ".join([f"{row.Mots_clés}:1" for _, row in df_keywords.iterrows()])
    st.markdown("#### Copiez-collez cette suggestion dans la zone mots-clés pondérés :")
    st.code(suggestion)

if uploaded_zip and keywords_input.strip():
    keywords_weights = parse_keywords(keywords_input)

    with zipfile.ZipFile(uploaded_zip) as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]

        if not pdf_files:
            st.error("Aucun fichier PDF trouvé dans le ZIP.")
        else:
            results = []
            texts = {}

            for pdf_name in pdf_files:
                with z.open(pdf_name) as pdf_file:
                    pdf_bytes = pdf_file.read()
                    text = extract_text_from_pdf_bytes(pdf_bytes)
                    texts[pdf_name] = text

                    score = score_text(text, keywords_weights)
                    score_max = sum(keywords_weights.values())
                    match_pct = min(100.0, round((score / score_max) * 100, 1))

                    results.append((clean_filename(pdf_name), score, match_pct))

            # Tri décroissant sur le score
            results.sort(key=lambda x: x[1], reverse=True)

            # Résultats globaux
            st.write("### Résultats du scoring")
            df_results = pd.DataFrame(results, columns=["Candidat", "Score brut", "Correspondance (%)"])
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Top X
            df_top = df_results.nlargest(top, "Correspondance (%)")

            # --- Graphique : Correspondance en % des top candidats ---
            bar_scores = alt.Chart(df_top).mark_bar().encode(
                x=alt.X('Candidat', sort='-y'),
                y=alt.Y('Correspondance (%):Q', title='Score de correspondance (%)'),
                tooltip=['Candidat', 'Correspondance (%)']
            ).properties(title=f"Scores de correspondance des Top {top} candidats")
            st.altair_chart(bar_scores, use_container_width=True)

            # --- Occurrences totales des mots-clés dans les Top candidats ---
            total_counts = Counter()
            for candidate in df_top['Candidat']:
                for filename, text in texts.items():
                    if clean_filename(filename) == candidate:
                        total_counts += count_keywords_in_text(text, keywords_weights)

            df_counts = pd.DataFrame(total_counts.items(), columns=['Mot-clé', 'Occurrences']).sort_values(by='Occurrences', ascending=False)

            bar_counts = alt.Chart(df_counts).mark_bar().encode(
                x=alt.X('Mot-clé', sort='-y'),
                y='Occurrences',
                tooltip=['Mot-clé', 'Occurrences']
            ).properties(title=f"Occurrences des mots-clés dans les Top {top} CV")
            st.altair_chart(bar_counts, use_container_width=True)

            # --- Diagramme groupé : Occurrences des mots-clés par candidat ---
            data_occurrences = []
            for candidate in df_top['Candidat']:
                for filename, text in texts.items():
                    if clean_filename(filename) == candidate:
                        counts = count_keywords_in_text(text, keywords_weights)
                        for kw, occ in counts.items():
                            data_occurrences.append({"Candidat": candidate, "Mot-clé": kw, "Occurrences": occ})

            df_occurrences = pd.DataFrame(data_occurrences)

            grouped_chart = alt.Chart(df_occurrences).mark_bar().encode(
                x=alt.X('Candidat:N', title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Occurrences:Q', title='Nombre d\'occurrences'),
                color=alt.Color('Mot-clé:N'),
                column=alt.Column('Mot-clé:N', header=alt.Header(labelOrient='bottom', labelAngle=0, labelFontSize=18))
            ).properties(
                width=100,
                height=250,
                title=f"Occurrences des mots-clés par candidat dans le Top {top}"
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            ).configure_title(
                fontSize=16,
                anchor='start'
            )

            st.altair_chart(grouped_chart, use_container_width=True)
else:
    if uploaded_zip and not keywords_input.strip():
        st.warning("Merci d'entrer les mots-clés avec leurs poids.")
