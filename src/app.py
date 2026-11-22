# --- Bibliothèque standard ---
import io
import os
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

# --- Bibliothèques tierces ---
import altair as alt
import docx  # python-docx
import nltk
import pandas as pd
import pymupdf
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Données NLTK ---
@st.cache_resource
def ensure_nltk_data() -> None:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


ensure_nltk_data()


# --- Extraction texte (PDF / Word), mise en cache pour les reruns Streamlit ---
SUPPORTED_EXT = (".pdf", ".docx")


def _extract_pdf(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        # sort=True : meilleur ordre de lecture (CV sur plusieurs colonnes)
        return "\n".join(page.get_text(sort=True) for page in doc)


def _extract_docx(data: bytes) -> str:
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs]
    # Les CV Word utilisent souvent des tableaux pour la mise en page
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


@st.cache_data(show_spinner=False)
def extract_text(data: bytes, ext: str) -> str:
    return _extract_pdf(data) if ext == ".pdf" else _extract_docx(data)


# --- Normalisation pour le matching : minuscules, sans accents ---
# On garde chiffres, "+" et "#" pour ne pas casser "python3", "c++", "c#".
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- Parsing mots-clés pondérés ---
def parse_keywords(text: str) -> dict:
    kw = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.rsplit(":", 1)
            try:
                weight = float(v.strip().replace(",", "."))
            except ValueError:
                weight = 1.0
        else:
            k, weight = part, 1.0
        k = normalize(k)
        if k:
            kw[k] = weight
    return kw


# --- Comptage des occurrences (mots entiers ou expressions) ---
def count_keywords(norm_text: str, keywords: dict) -> dict:
    counts = {}
    for kw in keywords:
        pattern = r"(?<![a-z0-9+#])" + re.escape(kw) + r"(?![a-z0-9+#])"
        counts[kw] = len(re.findall(pattern, norm_text))
    return counts


def score_from_counts(counts: dict, keywords: dict) -> float:
    return sum(counts[kw] * w for kw, w in keywords.items())


# --- Nettoyage du nom de fichier en nom propre ---
def clean_filename(filepath):
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    if name_without_ext.lower().startswith("cv_"):
        name_without_ext = name_without_ext[3:]
    name_parts = name_without_ext.replace("-", " ").replace("_", " ").split()
    return " ".join(part.capitalize() for part in name_parts)


def clean_text(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords(text, top_k=10):
    cleaned_text = clean_text(text)
    french_stopwords = stopwords.words("french")

    vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=100)
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    top_terms = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_k]
    max_score = top_terms[0][1] if top_terms else 1

    col = "Score de suggestions (en %)"
    df = pd.DataFrame(top_terms, columns=["Mots_clés", col])
    df[col] = ((df[col] / max_score) * 100).round(2)
    return df


# --- Chargement des documents : liste de (nom, bytes) ---
def load_documents(folder: str, zip_file, recurse: bool):
    docs, ignored = [], []
    if zip_file is not None:
        with zipfile.ZipFile(zip_file) as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if not base or base.startswith(("~$", ".")):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in SUPPORTED_EXT:
                    docs.append((name, z.read(name)))
                elif ext == ".doc":
                    ignored.append(name)
    elif folder.strip():
        root = Path(folder.strip().strip('"')).expanduser()
        if not root.is_dir():
            return None, ignored
        files = root.rglob("*") if recurse else root.glob("*")
        for p in sorted(files):
            if not p.is_file() or p.name.startswith(("~$", ".")):
                continue
            ext = p.suffix.lower()
            if ext in SUPPORTED_EXT:
                docs.append((str(p.relative_to(root)), p.read_bytes()))
            elif ext == ".doc":
                ignored.append(p.name)
    return docs, ignored


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

source = st.radio("Source des CV", ["📂 Dossier sur mon ordinateur", "🗜️ Fichier ZIP"], horizontal=True)

uploaded_zip = None
folder_path = ""
recursive = True
if source.startswith("📂"):
    folder_path = st.text_input(
        "Chemin du dossier contenant les CV (PDF et Word .docx)",
        placeholder=r"C:\Users\moi\Documents\CV   ou   /Users/moi/Documents/CV",
    )
    recursive = st.checkbox("Inclure les sous-dossiers", value=True)
    st.caption("Fonctionne si l'application tourne sur votre ordinateur (`streamlit run app.py`).")
else:
    uploaded_zip = st.file_uploader("Téléversez le ZIP des CV (PDF et Word .docx)", type=["zip"])

job_description = st.text_area("Collez votre offre d'emploi ici (texte brut)", height=250)

if job_description:
    top_k = st.number_input("Nombre de mots-clés à extraire de l'offre d'emploi", min_value=1, max_value=50, value=10)

keywords_input = st.text_input("Mots-clés et poids (ex: python:3, docker:2, sql)")

top = st.number_input("Sélectionnez votre top des candidats à analyser", min_value=1, max_value=100, value=10)

if job_description:
    df_keywords = extract_keywords(job_description, top_k)
    st.markdown("### Mots-clés suggérés")
    st.dataframe(df_keywords, use_container_width=True, hide_index=True)

    suggestion = ", ".join(f"{kw}:1" for kw in df_keywords["Mots_clés"])
    st.markdown("#### Copiez-collez cette suggestion dans la zone mots-clés pondérés :")
    st.code(suggestion)

has_source = uploaded_zip is not None or bool(folder_path.strip())

if has_source and keywords_input.strip():
    keywords_weights = parse_keywords(keywords_input)
    score_max = sum(keywords_weights.values())
    documents, ignored = load_documents(folder_path, uploaded_zip, recursive)

    if documents is None:
        st.error("Dossier introuvable. Vérifiez le chemin.")
    elif not keywords_weights:
        st.warning("Aucun mot-clé valide détecté.")
    elif score_max <= 0:
        st.warning("La somme des poids doit être strictement positive.")
    elif not documents:
        st.error("Aucun fichier PDF ou Word (.docx) trouvé.")
    else:
        if ignored:
            st.info(f"{len(ignored)} fichier(s) .doc ignoré(s) (ancien format Word) : enregistrez-les en .docx.")

        rows = []         # un dict par CV
        occurrences = []  # format long : candidat / mot-clé / occurrences
        seen = Counter()

        for name, data in documents:
            candidate = clean_filename(name)
            seen[candidate] += 1
            if seen[candidate] > 1:  # même nom dans deux sous-dossiers
                candidate = f"{candidate} ({seen[candidate]})"
            try:
                text = extract_text(data, os.path.splitext(name)[1].lower())
            except Exception:
                st.warning(f"Impossible de lire « {name} » (fichier ignoré).")
                continue

            if not text.strip():
                st.warning(f"« {name} » ne contient pas de texte (PDF scanné ? OCR nécessaire).")

            counts = count_keywords(normalize(text), keywords_weights)
            score = score_from_counts(counts, keywords_weights)
            match_pct = min(100.0, round(score / score_max * 100, 1))

            rows.append({"Candidat": candidate, "Score brut": score, "Correspondance (%)": match_pct})
            for kw, occ in counts.items():
                occurrences.append({"Candidat": candidate, "Mot-clé": kw, "Occurrences": occ})

        if not rows:
            st.error("Aucun CV exploitable.")
        else:
            df_results = (
                pd.DataFrame(rows)
                .sort_values("Score brut", ascending=False)
                .reset_index(drop=True)
            )
            st.write("### Résultats du scoring")
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            df_top = df_results.head(top)
            df_occ = pd.DataFrame(occurrences)
            df_occ_top = df_occ[df_occ["Candidat"].isin(df_top["Candidat"])]

            # --- Correspondance (%) des top candidats ---
            bar_scores = alt.Chart(df_top).mark_bar().encode(
                x=alt.X("Candidat:N", sort="-y"),
                y=alt.Y("Correspondance (%):Q", title="Score de correspondance (%)"),
                tooltip=["Candidat", "Correspondance (%)"],
            ).properties(title=f"Scores de correspondance des Top {top} candidats")
            st.altair_chart(bar_scores, use_container_width=True)

            # --- Occurrences totales des mots-clés dans le top ---
            df_counts = (
                df_occ_top.groupby("Mot-clé", as_index=False)["Occurrences"].sum()
                .sort_values("Occurrences", ascending=False)
            )
            bar_counts = alt.Chart(df_counts).mark_bar().encode(
                x=alt.X("Mot-clé:N", sort="-y"),
                y="Occurrences:Q",
                tooltip=["Mot-clé", "Occurrences"],
            ).properties(title=f"Occurrences des mots-clés dans les Top {top} CV")
            st.altair_chart(bar_counts, use_container_width=True)

            # --- Diagramme groupé : occurrences par candidat ---
            grouped_chart = alt.Chart(df_occ_top).mark_bar().encode(
                x=alt.X("Candidat:N", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Occurrences:Q", title="Nombre d'occurrences"),
                color=alt.Color("Mot-clé:N"),
                column=alt.Column("Mot-clé:N", header=alt.Header(labelOrient="bottom", labelAngle=0, labelFontSize=18)),
            ).properties(
                width=100,
                height=250,
                title=f"Occurrences des mots-clés par candidat dans le Top {top}",
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14,
            ).configure_title(
                fontSize=16,
                anchor="start",
            )
            st.altair_chart(grouped_chart, use_container_width=False)
elif has_source and not keywords_input.strip():
    st.warning("Merci d'entrer les mots-clés avec leurs poids.")
