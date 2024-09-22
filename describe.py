import pandas as pd
import math
import numpy as np
import sys
import os

def length(colonne):
    colonne = colonne.astype('float')
    colonne = colonne[~np.isnan(colonne)]
    return len(colonne)

def moyenne(colonne):
    if colonne.count() == 0 or colonne.dropna().empty:
        return np.nan
    moyenne = colonne.sum() / colonne.count()
    return moyenne

def ecart_type(colonne):
    mean = moyenne(colonne)
    valid_values = colonne.dropna()
    variance = sum((x - mean) ** 2 for x in valid_values) / (len(valid_values) - 1)
    std_dev = math.sqrt(variance)
    return std_dev

def minimum(colonne):
    res = colonne[0]
    for x in colonne:
        if res > x:
            res = x
    return res

def maximum(colonne):
    res = colonne[0]
    for x in colonne:
        if res < x:
            res = x
    return res

def q1(colonne):
    sorted_data = sorted(colonne)
    n = (len(sorted_data) - 1) * 0.25
    floor = np.floor(n)
    cell = np.ceil(n)
    if floor == cell:
        return sorted_data[int(n)]
    else:
        d0 = sorted_data[int(floor)] * (cell - n)
        d1 = sorted_data[int(cell)] * (n - floor)
        return d0 + d1

def q3(colonne):
    sorted_data = sorted(colonne)
    n = (len(sorted_data) - 1) * 0.75
    floor = np.floor(n)
    cell = np.ceil(n)
    if floor == cell:
        return sorted_data[int(n)]
    else:
        d0 = sorted_data[int(floor)] * (cell - n)
        d1 = sorted_data[int(cell)] * (n - floor)
        return d0 + d1
    
def mediane(colonne):
    sorted_data = sorted(colonne)
    n = len(sorted_data)
    res = 0
    if (n % 2 == 1):
        res = sorted_data[int((n+1) / 2) - 1]
    else:
        pos1 = (n // 2) - 1
        pos2 = n // 2
        res = (sorted_data[pos1] + sorted_data[pos2]) / 2
    return res

def describe(df):
    # Sélectionner uniquement les colonnes numériques
    df = df.select_dtypes(include='number')
    # Calcul des statistiques
    stats = {
        'Count': [length(df[col]) for col in df.columns],
        'Mean': [moyenne(df[col]) for col in df.columns],
        'Std': [ecart_type(df[col]) for col in df.columns],
        'Min': [minimum(df[col]) for col in df.columns],
        '25%': [q1(df[col]) for col in df.columns],
        '50%': [mediane(df[col]) for col in df.columns],
        '75%': [q3(df[col]) for col in df.columns],
        'Max': [maximum(df[col]) for col in df.columns],
    }

    # Création d'un DataFrame à partir des statistiques
    stats_df = pd.DataFrame(stats, index=df.columns)
    stats_df = stats_df.dropna()
    # Affichage du DataFrame
    print(stats_df.T)

def main():
    # Vérification des arguments de ligne de commande
    if len(sys.argv) < 2:
        print("Erreur : vous devez spécifier le chemin du fichier CSV.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Vérification si le fichier existe
    if not os.path.exists(file_path):
        print(f"Erreur : le fichier '{file_path}' n'existe pas.")
        sys.exit(1)
    
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(file_path)
        
        # Vérification si le fichier est vide
        if df.empty:
            print("Erreur : le fichier est vide.")
            sys.exit(1)
        
        # Affichage des statistiques avec pandas et la fonction personnalisée
        print("describe original")
        print(df.describe())
        print("mes valeurs")
        describe(df)
    
    except pd.errors.EmptyDataError:
        print("Erreur : le fichier est vide ou illisible.")
    except pd.errors.ParserError:
        print("Erreur : erreur de parsing du fichier.")
    except PermissionError:
        print(f"Erreur : vous n'avez pas les droits nécessaires pour accéder au fichier '{file_path}'.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {str(e)}")

if __name__ == "__main__":
    main()