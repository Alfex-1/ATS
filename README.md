# 📊 ATS — Applicant Tracking System

ATS est une application légère de tri et d’analyse de CV basée sur la correspondance textuelle avec une offre d’emploi.

Elle permet d’évaluer rapidement plusieurs candidats à partir de mots-clés pondérés, et de visualiser leur adéquation avec un poste.

---

## 🚀 Fonctionnalités

### 🔎 Scoring des candidats
- Calcul automatique d’un score de correspondance entre chaque CV et l’offre d’emploi
- Pondération possible des mots-clés selon leur importance métier
- Classement des candidats du plus pertinent au moins pertinent

### 📊 Dashboard analytique
- Visualisation des résultats de matching
- Analyse des occurrences de mots-clés par candidat
- Comparaison rapide entre profils

### 🧠 Assistant de mots-clés (bonus)
- Extraction automatique de mots-clés depuis une offre d’emploi
- Suggestions ajustables selon vos besoins
- Attribution de poids personnalisables pour affiner le scoring

---

## 🎯 Objectif

Réduire le temps perdu à lire des CV non pertinents en automatisant une première couche de filtrage textuel.

---

## ⚙️ Stack technique

- Python
- Streamlit
- scikit-learn (TF-IDF)
- pandas
- nltk

---

## 📦 Utilisation

1. Coller ou importer les CV
2. Ajouter une offre d’emploi
3. Générer ou ajuster les mots-clés
4. Lancer l’analyse
5. Explorer le dashboard de matching

---

## 🌐 Application

👉 https://ats-hr-assistant.streamlit.app/

---

## ⚠️ Limites

- Analyse purement textuelle (pas de compréhension sémantique avancée)
- Sensible à la qualité des CV (format, structure, langage)
- Ne remplace pas une évaluation humaine — elle la prépare

---

## 🧠 Vision

Cet outil n’est pas un recruteur automatique.  
C’est un filtre intelligent qui évite de perdre du temps sur les mauvais profils.