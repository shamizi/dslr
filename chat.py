import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta, epsilon=1e-10):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    cost = (-1/m) * (y.T.dot(np.log(predictions)) + (1 - y).T.dot(np.log(1 - predictions)))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations, lambda_=0.1):
    """Algorithme de descente de gradient avec régularisation L2"""
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        predictions = sigmoid(X.dot(theta))
        errors = predictions - y
        regularization_term = (lambda_ / m) * theta  # Terme de régularisation
        theta -= (learning_rate / m) * (X.T.dot(errors) + regularization_term)
        cost_history[i] = compute_cost(X, y, theta) + (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))

    return theta, cost_history

def predict(X, all_theta):
    """Faire des prédictions en utilisant le modèle one-vs-all"""
    probabilities = sigmoid(X.dot(all_theta))
    return np.argmax(probabilities, axis=1)

def main():
    # Charger les données
    train_df = pd.read_csv('datasets/dataset_train.csv')
    train_df = train_df.dropna()

    houses = train_df['Hogwarts House'].astype('category').cat.categories
    train_df['Hogwarts House'] = train_df['Hogwarts House'].astype('category').cat.codes

    selected_features = ['Defense Against the Dark Arts', 'Astronomy', 'Flying', 'Potions', 'Transfiguration', 'Divination']
    X_train = train_df[selected_features].values
    y_train = train_df['Hogwarts House'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))

    # Les meilleurs hyperparamètres (issus de la validation croisée)
    best_learning_rate = 0.01
    best_num_iterations = 5000
    best_lambda = 0.001

    print(f"Entraînement avec learning_rate={best_learning_rate}, num_iterations={best_num_iterations}, lambda={best_lambda}")

    # Entraînement final avec les meilleurs hyperparamètres
    num_classes = y_train_one_hot.shape[1]
    num_features = X_train.shape[1]
    all_theta = np.zeros((num_features, num_classes))

    for i in range(num_classes):
        y_i = y_train_one_hot[:, i]
        theta_i = np.zeros(num_features)
        theta_i, _ = gradient_descent(X_train, y_i, theta_i, best_learning_rate, best_num_iterations, best_lambda)
        all_theta[:, i] = theta_i

    joblib.dump({
        'theta': all_theta,
        'encoder': encoder,
        'scaler': scaler,
        'selected_features': selected_features,
        'houses': houses
    }, 'model.pkl')

    print("Modèle sauvegardé dans model.pkl")

    # Faire des prédictions sur dataset_train.csv pour mesurer l'accuracy
    y_pred = predict(X_train, all_theta)

    # Calculer et afficher l'accuracy
    accuracy = accuracy_score(y_train, y_pred)
    print(f'Accuracy sur dataset_train.csv : {accuracy:.3f}')

if __name__ == "__main__":
    main()
