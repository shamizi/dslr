import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        predictions = sigmoid(X.dot(theta))
        errors = predictions - y
        theta -= (learning_rate / m) * X.T.dot(errors)
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

def predict(X, all_theta):
    """Fonction de prédiction utilisant le modèle one-vs-all"""
    probabilities = sigmoid(X.dot(all_theta))
    return np.argmax(probabilities, axis=1)

def validate_hyperparameters(X_train, y_train_one_hot):
    best_accuracy = 0
    best_learning_rate = 0
    best_num_iterations = 0
    learning_rates = [0.001, 0.01, 0.1]
    num_iterations_list = [1000, 5000, 10000]

    for learning_rate in learning_rates:
        for num_iterations in num_iterations_list:
            print(f'Validation avec learning_rate={learning_rate}, num_iterations={num_iterations}')
            num_classes = y_train_one_hot.shape[1]
            num_features = X_train.shape[1]
            all_theta = np.zeros((num_features, num_classes))

            for i in range(num_classes):
                y_i = y_train_one_hot[:, i]
                theta_i = np.zeros(num_features)
                theta_i, _ = gradient_descent(X_train, y_i, theta_i, learning_rate, num_iterations)
                all_theta[:, i] = theta_i

            # Évaluer le modèle sur un ensemble de validation
            X_val, X_test, y_val, y_test = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=42)
            y_pred = predict(X_val, all_theta)
            y_val_labels = np.argmax(y_val, axis=1)
            accuracy = accuracy_score(y_val_labels, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = learning_rate
                best_num_iterations = num_iterations

    return best_learning_rate, best_num_iterations, best_accuracy

def main():
    # Charger les données
    train_df = pd.read_csv('C:/Users/said/Desktop/choixpeau/datasets/dataset_train.csv')
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

    best_learning_rate, best_num_iterations, best_accuracy = validate_hyperparameters(X_train, y_train_one_hot)

    print(f"Meilleur taux d'apprentissage: {best_learning_rate}")
    print(f"Meilleur nombre d'itérations: {best_num_iterations}")
    print(f"Meilleure précision: {best_accuracy}")

    # Entraînement final avec les meilleurs hyperparamètres
    num_classes = y_train_one_hot.shape[1]
    num_features = X_train.shape[1]
    all_theta = np.zeros((num_features, num_classes))

    for i in range(num_classes):
        y_i = y_train_one_hot[:, i]
        theta_i = np.zeros(num_features)
        theta_i, _ = gradient_descent(X_train, y_i, theta_i, best_learning_rate, best_num_iterations)
        all_theta[:, i] = theta_i

    joblib.dump({
        'theta': all_theta,
        'encoder': encoder,
        'scaler': scaler,
        'selected_features': selected_features,
        'houses': houses
    }, 'model.pkl')

    print("Modèle sauvegardé dans model.pkl")

    #### test sur donnée ou on connait la réponse
    test_df = pd.read_csv('C:/Users/said/Desktop/choixpeau/datasets/dataset_train.csv')
    if 'Hogwarts House' in test_df.columns:
        test_df['Hogwarts House'] = test_df['Hogwarts House'].astype('category').cat.codes
        X_test = test_df[selected_features].values
        y_test = test_df['Hogwarts House'].values

        X_test = scaler.transform(X_test)
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        y_pred = predict(X_test, all_theta)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy sur le fichier de test: {accuracy:.3f}')

if __name__ == "__main__":
    main()
