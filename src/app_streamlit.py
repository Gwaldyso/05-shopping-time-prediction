import streamlit as st
import pandas as pd
import joblib
import os

import pdfplumber
from PIL import Image
import pytesseract

MODEL_PATH = os.path.join("models", "shopping_time_model.joblib")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Lance d'abord python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def setup_tesseract_if_needed():
    """Configure le chemin de Tesseract si nécessaire (Windows)."""
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



def parse_shopping_list_text(text: str):
    text = text.lower()

    categories = {
        "alimentaire": [
            # anciens mots
            "yaourt", "yaourts", "pomme", "riz", "pates", "pâtes", "lait", "pain", "fromage", "steak",
            # viandes
            "boeuf", "bœuf", "poulet", "cuisse", "cuisses", "blanc de poulet", "côte", "cotis",
            # poissons
            "poisson", "bar", "tilapia",
            # féculents / accompagnements
            "riz", "riz cassé", "macedoine", "macédoine", "mais", "maïs",
            # légumes / condiments
            "tomate", "tomates", "poivron", "ail", "sauce",
            # autres
            "huile"
        ],
        "vetements": ["jean", "jeans", "robe", "tshirt", "t-shirt", "chemise", "pantalon"],
        "electronique": ["tv", "télé", "tele", "télévision", "telephone", "smartphone", "ordinateur", "laptop", "tablette"],
        "maison": ["coussin", "assiette", "verre", "rideau", "linge", "poele", "poêle", "casserole"],
        "beaute": ["shampoing", "shampooing", "gel douche", "parfum", "maquillage", "creme", "crème"],
        "sport": ["ballon", "chaussures de sport", "dumbbell", "tapis yoga"],
        "librairie": ["livre", "cahier", "stylo", "agenda"],
    }

    counts = {
        "items_alimentaire": 0,
        "items_vetements": 0,
        "items_electronique": 0,
        "items_maison": 0,
        "items_beaute": 0,
        "items_sport": 0,
        "items_librairie": 0,
    }

    for cat, keywords in categories.items():
        for kw in keywords:
            if kw in text:
                if cat == "alimentaire":
                    counts["items_alimentaire"] += 1
                elif cat == "vetements":
                    counts["items_vetements"] += 1
                elif cat == "electronique":
                    counts["items_electronique"] += 1
                elif cat == "maison":
                    counts["items_maison"] += 1
                elif cat == "beaute":
                    counts["items_beaute"] += 1
                elif cat == "sport":
                    counts["items_sport"] += 1
                elif cat == "librairie":
                    counts["items_librairie"] += 1

    return counts



def extract_text_from_uploaded_file(uploaded_file):
    """
    Gère TXT, CSV, PDF, image (PNG/JPG) et renvoie une chaîne de texte.
    """
    filename = uploaded_file.name.lower()

    # TXT
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    # CSV
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        # On prend toutes les colonnes et toutes les lignes, concaténées
        text = " ".join(df.astype(str).values.ravel().tolist())
        return text

    # PDF
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += "\n" + page_text
        return text

    # Images : png, jpg, jpeg
    if filename.endswith((".png", ".jpg", ".jpeg")):
        setup_tesseract_if_needed()
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image, lang="fra+eng")  # français + anglais
        return text

    # Si format non géré :
    raise ValueError("Format de fichier non supporté. Utilise txt, csv, pdf, png, jpg ou jpeg.")


def main():
    st.title("🛒 Prédiction du temps de shopping")
    st.write("Estime le temps passé en magasin en fonction du profil client et de sa liste de courses.")

    model = load_model()

    # =========================
    # Sidebar : infos client
    # =========================
    with st.sidebar:
        st.header("🧍 Profil client")
        age = st.number_input("Âge", min_value=10, max_value=100, value=30)
        gender = st.selectbox("Sexe", ["femme", "homme"])
        profile = st.selectbox("Profil", ["rapide", "normal", "flaneur", "methodique"])
        store_type = st.selectbox(
            "Type de magasin",
            ["supermarche", "hypermarche", "centre_commercial", "boutique"],
        )

        st.header("🗓 Période")
        day_label = st.selectbox(
            "Jour de la semaine",
            ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
        )
        day_map = {
            "lundi": 0,
            "mardi": 1,
            "mercredi": 2,
            "jeudi": 3,
            "vendredi": 4,
            "samedi": 5,
            "dimanche": 6,
        }
        day_of_week = day_map[day_label]
        period = "semaine" if day_of_week < 5 else "weekend"

        special_event = st.selectbox(
            "Événement spécial",
            [
                "aucun",
                "soldes_ete",
                "soldes_hiver",
                "noel",
                "paques",
                "rentree",
                "black_friday",
                "fin_annee",
            ],
        )

        hour = st.slider("Heure de la visite", min_value=0, max_value=23, value=16)
        has_shopping_list = 1 if st.checkbox("Le client a une liste de courses", value=True) else 0

    # =========================
    # Liste de courses
    # =========================
    st.header("🧾 Liste de courses")

    mode = st.radio(
        "Mode de saisie de la liste :",
        [
            "Saisie manuelle (par catégories)",
            "Texte libre",
            "Importer un fichier (txt/csv/pdf/image)",
        ],
        index=0,
    )

    # Valeurs par défaut
    counts = {
        "items_alimentaire": 0,
        "items_vetements": 0,
        "items_electronique": 0,
        "items_maison": 0,
        "items_beaute": 0,
        "items_sport": 0,
        "items_librairie": 0,
    }

    if mode == "Texte libre":
        raw_list = st.text_area(
            "Liste de courses (ex: '10 yaourts, 2 jeans, 1 TV, 3 shampoings')",
            "",
        )
        if raw_list.strip():
            counts = parse_shopping_list_text(raw_list)
            st.write("Approximation des catégories : ", counts)

    elif mode == "Saisie manuelle (par catégories)":
        col1, col2 = st.columns(2)
        with col1:
            items_alimentaire = st.number_input("Articles alimentaires", min_value=0, value=10)
            items_vetements = st.number_input("Articles vêtements", min_value=0, value=2)
            items_electronique = st.number_input("Articles électroniques", min_value=0, value=0)
            items_maison = st.number_input("Articles maison/jardin", min_value=0, value=2)
        with col2:
            items_beaute = st.number_input("Articles beauté/santé", min_value=0, value=1)
            items_sport = st.number_input("Articles sport/loisirs", min_value=0, value=0)
            items_librairie = st.number_input("Articles librairie/papeterie", min_value=0, value=0)

        counts = {
            "items_alimentaire": items_alimentaire,
            "items_vetements": items_vetements,
            "items_electronique": items_electronique,
            "items_maison": items_maison,
            "items_beaute": items_beaute,
            "items_sport": items_sport,
            "items_librairie": items_librairie,
        }

    elif mode == "Importer un fichier (txt/csv/pdf/image)":
        uploaded_file = st.file_uploader(
            "Importer un fichier de liste de courses (.txt, .csv, .pdf, .png, .jpg, .jpeg)",
            type=["txt", "csv", "pdf", "png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            try:
                text = extract_text_from_uploaded_file(uploaded_file)
                st.text_area("Texte extrait :", text, height=150)
                counts = parse_shopping_list_text(text)
                st.write("Approximation des catégories à partir du fichier : ", counts)
            except Exception as e:
                st.error(f"Erreur lors de la lecture/du traitement du fichier : {e}")

    total_items = sum(counts.values())
    nb_categories = sum(v > 0 for v in counts.values())

    is_weekend = 1 if period == "weekend" else 0
    is_sales = 1 if special_event in ["soldes_ete", "soldes_hiver", "black_friday"] else 0
    is_holiday = 1 if special_event in ["noel", "paques", "rentree", "fin_annee"] else 0

    # =========================
    # Construction de l'input
    # =========================
    input_dict = {
        "age": [age],
        "gender": [gender],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "total_items": [total_items],
        "nb_categories": [nb_categories],
        "items_alimentaire": [counts["items_alimentaire"]],
        "items_vetements": [counts["items_vetements"]],
        "items_electronique": [counts["items_electronique"]],
        "items_maison": [counts["items_maison"]],
        "items_beaute": [counts["items_beaute"]],
        "items_sport": [counts["items_sport"]],
        "items_librairie": [counts["items_librairie"]],
        "is_weekend": [is_weekend],
        "is_sales": [is_sales],
        "is_holiday": [is_holiday],
        "has_shopping_list": [has_shopping_list],
        "profile": [profile],
        "store_type": [store_type],
        "period": [period],
        "special_event": [special_event],
    }

    # =========================
    # Prédiction
    # =========================
    if st.button("Prédire le temps de shopping"):
        X = pd.DataFrame(input_dict)
        y_pred = model.predict(X)[0]
        st.subheader(f"⏱ Temps estimé : {y_pred:.1f} minutes")


if __name__ == "__main__":
    main()
