import joblib
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def load_and_preprocess_dataset(filename):
    data, meta = arff.loadarff(filename)

    data_array = np.array(data.tolist(), dtype=np.float32)

    X = data_array[:, :-1]
    y = data_array[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


X_train, X_test, y_train, y_test = load_and_preprocess_dataset('mnist_784.arff')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

joblib.dump(knn, 'model.joblib')

# Оценка точности классификатора на тестовой выборке
accuracy = knn.score(X_test, y_test)
print(f'Точность классификатора k-NN: {accuracy:.2f}')
