import pandas as pd
import matplotlib.pyplot as plt

def lire_et_visualiser_dataframe(fichier_csv):
    df = pd.read_csv(fichier_csv)

    colonnes = df.columns
    for col in colonnes:
        plt.plot(df[col])
        plt.title(col)
        plt.xlabel('Index')
        plt.show()

    print(df)

# Appel de la fonction pour lire et visualiser le DataFrame
lire_et_visualiser_dataframe('resultats_flot_optique.csv')
