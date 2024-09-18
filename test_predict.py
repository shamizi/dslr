import numpy as np
import pandas as pd
import joblib  # Pour charger le modèle entraîné
from sklearn.impute import SimpleImputer

def sigmoid(z):
    """Fonction sigmoïde"""
    return 1 / (1 + np.exp(-z))

def predict(X, all_theta):
    """Faire des prédictions en utilisant le modèle entraîné"""
    probabilities = sigmoid(X.dot(all_theta))
    return np.argmax(probabilities, axis=1)

def main():
    # Charger le modèle entraîné
    model = joblib.load('model.pkl')
    all_theta = model['theta']
    encoder = model['encoder']
    scaler = model['scaler']
    selected_features = model['selected_features']
    houses = model['houses']  # Récupérer les noms des maisons

    # Charger les données de test
    test_df = pd.read_csv('datasets/dataset_test.csv')

    # Vérifier si toutes les features sélectionnées sont présentes dans les données de test
    missing_features = set(selected_features) - set(test_df.columns)
    if missing_features:
        print(f"Attention : Les features suivantes sont manquantes dans les données de test: {missing_features}")
        return

    # Filtrer les colonnes selon les features sélectionnées
    X_test = test_df[selected_features]

    # Remplir les valeurs manquantes avec la moyenne des colonnes
    imputer = SimpleImputer(strategy='mean')
    X_test = imputer.fit_transform(X_test)

    # Normaliser les features de test avec le scaler entraîné
    X_test = scaler.transform(X_test)

    # Ajouter une colonne de 1 pour le biais
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    
    # Faire des prédictions sur le jeu de test
    y_pred = predict(X_test, all_theta)

    # Convertir les prédictions en noms de maisons
    predicted_houses = np.array(houses)[y_pred]

    # Créer un DataFrame avec les résultats et les noms des maisons prédits
    results = pd.DataFrame({
        'Index': np.arange(len(predicted_houses)),
        'Hogwarts House': predicted_houses
    })

    # Sauvegarder les résultats dans le fichier houses.csv
    results.to_csv('houses.csv', index=False)
    print("Predictions saved to houses.csv")

if __name__ == "__main__":
    main()
