import pandas as pd

def lire_et_visualiser_dataframe(fichier_csv):
    df = pd.read_csv(fichier_csv)
    print(df)

# Appel de la fonction pour lire et visualiser le DataFrame
lire_et_visualiser_dataframe('resultats_flot_optique.csv')
