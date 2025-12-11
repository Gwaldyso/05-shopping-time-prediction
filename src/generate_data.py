import os
import numpy as np
import pandas as pd

def generate_shopping_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # Profils de clients
    profiles = ["rapide", "normal", "flaneur", "methodique"]
    store_types = ["supermarche", "hypermarche", "centre_commercial", "boutique"]

    # Sexe / genre
    genders = ["femme", "homme"]
    gender = np.random.choice(genders, size=n_samples, p=[0.5, 0.5])

    # Jours de la semaine (0 = lundi, 6 = dimanche)
    day_of_week = np.random.randint(0, 7, size=n_samples)

    # Période semaine/weekend (0-4 semaine, 5-6 weekend)
    period = np.where(day_of_week < 5, "semaine", "weekend")

    # Événements spéciaux (fêtes, soldes, fin d’année…)
    special_events = [
        "aucun",
        "soldes_ete",
        "soldes_hiver",
        "noel",
        "paques",
        "rentree",
        "black_friday",
        "fin_annee",
    ]
    special_event = np.random.choice(
        special_events,
        size=n_samples,
        p=[0.55, 0.08, 0.07, 0.08, 0.05, 0.07, 0.05, 0.05],
    )

    age = np.random.randint(18, 70, size=n_samples)
    profile = np.random.choice(profiles, size=n_samples, p=[0.2, 0.5, 0.2, 0.1])
    store_type = np.random.choice(store_types, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

    # Heure
    hour = np.random.randint(9, 21, size=n_samples)  # 9h–20h

    # Flags à partir de period + special_event
    is_weekend = np.where(period == "weekend", 1, 0)
    is_sales = np.isin(
        special_event, ["soldes_ete", "soldes_hiver", "black_friday"]
    ).astype(int)
    is_holiday = np.isin(
        special_event, ["noel", "paques", "rentree", "fin_annee"]
    ).astype(int)

    # Liste de courses (par catégories)
    items_alimentaire = np.random.poisson(lam=10, size=n_samples).clip(0)
    items_vetements = np.random.poisson(lam=3, size=n_samples).clip(0)
    items_electronique = np.random.poisson(lam=1, size=n_samples).clip(0)
    items_maison = np.random.poisson(lam=2, size=n_samples).clip(0)
    items_beaute = np.random.poisson(lam=2, size=n_samples).clip(0)
    items_sport = np.random.poisson(lam=1, size=n_samples).clip(0)
    items_librairie = np.random.poisson(lam=1, size=n_samples).clip(0)

    total_items = (
        items_alimentaire
        + items_vetements
        + items_electronique
        + items_maison
        + items_beaute
        + items_sport
        + items_librairie
    )

    nb_categories = (
        (items_alimentaire > 0).astype(int)
        + (items_vetements > 0).astype(int)
        + (items_electronique > 0).astype(int)
        + (items_maison > 0).astype(int)
        + (items_beaute > 0).astype(int)
        + (items_sport > 0).astype(int)
        + (items_librairie > 0).astype(int)
    )

    has_shopping_list = np.random.binomial(1, 0.6, size=n_samples)

    # Base time en minutes (vecteur)
    base_time = 10.0
    time = np.full(n_samples, base_time, dtype=float)

    # Effet nombre d'articles + diversité
    time += total_items * np.random.uniform(1.2, 2.0)
    time += nb_categories * 3.0

    # Effet weekend / soldes / fêtes
    time += is_weekend * 10.0
    time += is_sales * 15.0
    time += is_holiday * 20.0

    # Profil client
    time += np.select(
        [
            profile == "rapide",
            profile == "normal",
            profile == "flaneur",
            profile == "methodique",
        ],
        [
            -10.0,
            0.0,
            15.0,
            5.0,
        ],
        default=0.0,
    )

    # Type de magasin
    time += np.select(
        [
            store_type == "supermarche",
            store_type == "hypermarche",
            store_type == "centre_commercial",
            store_type == "boutique",
        ],
        [
            0.0,
            10.0,
            20.0,
            -5.0,
        ],
        default=0.0,
    )

    # Heure de pointe (17h–19h)
    is_rush_hour = ((hour >= 17) & (hour <= 19)).astype(int)
    time += is_rush_hour * 5.0

    # Liste de courses : sans liste → plus long
    time += np.where(has_shopping_list == 1, -5.0, 5.0)

    # ⚠️ ICI : on choisit de ne pas mettre de biais fixe sur le genre
    # mais tu peux le faire si tu veux simuler une différence moyenne.

    # Bruit aléatoire
    noise = np.random.normal(loc=0.0, scale=8.0, size=n_samples)
    time = time + noise

    # Plancher à 5 minutes
    time = np.clip(time, 5, None)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "profile": profile,
            "store_type": store_type,
            "period": period,
            "special_event": special_event,
            "day_of_week": day_of_week,
            "hour": hour,
            "is_weekend": is_weekend,
            "is_sales": is_sales,
            "is_holiday": is_holiday,
            "total_items": total_items,
            "nb_categories": nb_categories,
            "has_shopping_list": has_shopping_list,
            "items_alimentaire": items_alimentaire,
            "items_vetements": items_vetements,
            "items_electronique": items_electronique,
            "items_maison": items_maison,
            "items_beaute": items_beaute,
            "items_sport": items_sport,
            "items_librairie": items_librairie,
            "shopping_time_min": time,
        }
    )
    return df


def main():
    print("Génération du dataset de temps de shopping...")
    df = generate_shopping_dataset(n_samples=5000, random_state=42)

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "shopping_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Dataset généré : {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
