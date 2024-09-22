import numpy as np
import pandas as pd
import joblib  # Pour charger le modèle entraîné
from sklearn.impute import SimpleImputer
import argparse
import os

def sigmoid(z):
    """Fonction sigmoïde"""
    return 1 / (1 + np.exp(-z))

def predict(X, all_theta):
    """Faire des prédictions en utilisant le modèle entraîné"""
    probabilities = sigmoid(X.dot(all_theta))
    return np.argmax(probabilities, axis=1)

def main(test_file_path):
    if not os.path.exists('model.pkl'):
            print("Erreur : Le fichier 'model.pkl' est introuvable.")
            return
    try:
        model = joblib.load('model.pkl')
        all_theta = model['theta']
        encoder = model['encoder']
        scaler = model['scaler']
        selected_features = model['selected_features']
        houses = model['houses']

        test_df = pd.read_csv(test_file_path)
        X_test = test_df[selected_features]
        imputer = SimpleImputer(strategy='mean')
        X_test = imputer.fit_transform(X_test)
        X_test = scaler.transform(X_test)
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        
        #prediction
        y_pred = predict(X_test, all_theta)
        predicted_houses = np.array(houses)[y_pred]
        results = pd.DataFrame({
            'Index': np.arange(len(predicted_houses)),
            'Hogwarts House': predicted_houses
        })
        results.to_csv('houses.csv', index=False)
        print("Prédictions sauvegardées dans houses.csv")

    except FileNotFoundError:
        print(f"Erreur : Le fichier {test_file_path} est introuvable.")
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier {test_file_path} est vide.")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de prédiction des maisons de Poudlard sur un fichier de test.")
    parser.add_argument('test_file_path', type=str, help="Chemin du fichier CSV à utiliser pour les prédictions.")
    args = parser.parse_args()
    main(args.test_file_path)
