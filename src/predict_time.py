import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "shopping_time_model.joblib")


# ==========================
# Chargement du modèle
# ==========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. "
            "Lance d'abord : python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


# ==========================
# Fonctions utilitaires
# ==========================
def ask_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Veuillez entrer une valeur >= {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Veuillez entrer une valeur <= {max_value}.")
                continue
            return value
        except ValueError:
            print("Veuillez entrer un nombre entier valide.")


def ask_choice(prompt, options):
    """
    options = liste de chaînes. On affiche un menu numéroté.
    Retourne la chaîne choisie.
    """
    while True:
        print(prompt)
        for i, opt in enumerate(options, start=1):
            print(f"  {i}. {opt}")
        choice = input("Votre choix (1, 2, ...) : ").strip()
        if not choice.isdigit():
            print("Veuillez entrer un numéro valide.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        else:
            print("Choix invalide, réessayez.")


def ask_yes_no(prompt):
    while True:
        value = input(prompt + " (o/n) : ").strip().lower()
        if value in ("o", "oui"):
            return 1
        if value in ("n", "non"):
            return 0
        print("Répondez par 'o' (oui) ou 'n' (non).")


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



# ==========================
# Construction de l'input utilisateur
# ==========================
def build_input_from_user():
    print("=== Configuration du profil de shopping ===")

    age = ask_int("Âge du client : ", min_value=10, max_value=100)

    gender = ask_choice(
        "Sexe du client :",
        ["femme", "homme"],
    )

    profile = ask_choice(
        "Profil du client :",
        ["rapide", "normal", "flaneur", "methodique"],
    )

    store_type = ask_choice(
        "Type de magasin :",
        ["supermarche", "hypermarche", "centre_commercial", "boutique"],
    )

    # Jour de la semaine
    day_label = ask_choice(
        "Jour de la semaine :",
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

    # Période (cohérente avec le dataset : semaine vs weekend)
    period = "semaine" if day_of_week < 5 else "weekend"

    # Type d'événement spécial (inclut fin d'année)
    special_event = ask_choice(
        "Type d'événement (fêtes/soldes) :",
        ["aucun", "soldes_ete", "soldes_hiver", "noel", "paques", "rentree", "black_friday", "fin_annee"],
    )

    hour = ask_int("Heure de la visite (0–23) : ", min_value=0, max_value=23)

    has_shopping_list = ask_yes_no("Le client a-t-il une liste de courses ?")

    # Flags à partir de period + special_event
    is_weekend = 1 if period == "weekend" else 0
    is_sales = 1 if special_event in ["soldes_ete", "soldes_hiver", "black_friday"] else 0
    is_holiday = 1 if special_event in ["noel", "paques", "rentree", "fin_annee"] else 0

    print("\nSouhaitez-vous saisir une liste de courses en texte ?")
    use_text_list = ask_yes_no("Utiliser une liste texte")

    if use_text_list:
        print("\nExemple : '10 yaourts, 2 jeans, 1 TV, 3 shampoings'")
        raw_list = input("Entrez votre liste de courses :\n> ")
        counts = parse_shopping_list_text(raw_list)
        items_alimentaire = counts["items_alimentaire"]
        items_vetements = counts["items_vetements"]
        items_electronique = counts["items_electronique"]
        items_maison = counts["items_maison"]
        items_beaute = counts["items_beaute"]
        items_sport = counts["items_sport"]
        items_librairie = counts["items_librairie"]
        print("\nApproximation des catégories à partir du texte :")
        print(counts)
    else:
        print("\n=== Liste de courses par catégorie ===")
        items_alimentaire = ask_int("Nombre d'articles alimentaires : ", min_value=0)
        items_vetements = ask_int("Nombre d'articles de vêtements : ", min_value=0)
        items_electronique = ask_int("Nombre d'articles électroniques : ", min_value=0)
        items_maison = ask_int("Nombre d'articles maison/jardin : ", min_value=0)
        items_beaute = ask_int("Nombre d'articles beauté/santé : ", min_value=0)
        items_sport = ask_int("Nombre d'articles sport/loisirs : ", min_value=0)
        items_librairie = ask_int("Nombre d'articles librairie/papeterie : ", min_value=0)

    # Agrégations
    total_items = (
        items_alimentaire
        + items_vetements
        + items_electronique
        + items_maison
        + items_beaute
        + items_sport
        + items_librairie
    )

    nb_categories = sum(
        x > 0
        for x in [
            items_alimentaire,
            items_vetements,
            items_electronique,
            items_maison,
            items_beaute,
            items_sport,
            items_librairie,
        ]
    )

    data = {
        "age": [age],
        "gender": [gender],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "total_items": [total_items],
        "nb_categories": [nb_categories],
        "items_alimentaire": [items_alimentaire],
        "items_vetements": [items_vetements],
        "items_electronique": [items_electronique],
        "items_maison": [items_maison],
        "items_beaute": [items_beaute],
        "items_sport": [items_sport],
        "items_librairie": [items_librairie],
        "is_weekend": [is_weekend],
        "is_sales": [is_sales],
        "is_holiday": [is_holiday],
        "has_shopping_list": [has_shopping_list],
        "profile": [profile],
        "store_type": [store_type],
        "period": [period],
        "special_event": [special_event],
    }

    return data


# ==========================
# Prédiction
# ==========================
def predict_shopping_time(input_dict):
    model = load_model()
    X = pd.DataFrame(input_dict)
    y_pred = model.predict(X)
    return float(y_pred[0])


def main():
    print("=== Prédiction du temps de shopping ===\n")
    try:
        input_data = build_input_from_user()
        predicted_time = predict_shopping_time(input_data)
        print("\n=== Résultat ===")
        print(f"Temps estimé pour ce profil et cette liste : {predicted_time:.1f} minutes")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print("Erreur lors de la prédiction :", e)


if __name__ == "__main__":
    main()
