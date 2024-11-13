import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

dump = 'model.dump'

# Импорт датасета
data = pd.read_csv(r'Video_Games.csv')

# Сохранение в У объёма продаж в Японии
Y = data.JP_Sales
Y = np.log(Y+1)

# Сохранение датасета без объёма продаж в Японии
data = data.drop(['JP_Sales'], axis = 1)
# Удаление названий игр
data = data.drop(['Name'], axis = 1)

# Список категориальных признаков
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Создание экземпляра энкодера
encoder = OneHotEncoder(sparse_output=False)

# Полученние значений после one hot encoding
one_hot_encoded = encoder.fit_transform(data[categorical_columns])

# Преобразование полученных значений в массив типа int
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns), dtype=int)   

# Конкатенаця полученных значений в результате one hot encoding
data = pd.concat([data, one_hot_df], axis=1)

# Удаляем оставшиеся категориальные признаки
data = data.drop(categorical_columns, axis=1)

# Заполняем пропущенные значения NaN указанным числом
data = data.fillna(10)
x = data

for col in ['Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count',
            'NA_Sales', 'EU_Sales', 'Other_Sales']:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

def train_test_split(x, Y, test_size = 0.2, random_state = None):
    if random_state is not None:
        np.random.seed(random_state)
    data = pd.concat([x, Y], axis = 1)
    shuffled_indices = np.random.permutation(len(Y))
    n_tests = int(test_size * len(Y))
    test_indices = shuffled_indices[:n_tests]
    train_indices = shuffled_indices[n_tests:]
    data_train, data_test = data.iloc[train_indices, :], data.iloc[test_indices, :]
    return data_train.iloc[:, :-1], data_test.iloc[:, :-1], data_train.iloc[:, -1], data_test.iloc[:, -1]

# Класс регрессионной модели
class LinearRegressionGD(object):
    def __init__(self, lr = 0.001, max_iter = 300, tol = 1e-6, intercept = True):
        self.learning_rate = lr
        self.max_iteration = max_iter
        self.tolerance_convergence = tol
        self.intercept = intercept

    def fit(self, x, Y):
        self.x = x.copy()
        self.Y = Y.copy()
        if self.intercept:
            self.x = np.hstack((np.ones((self.x.shape[0], 1)), self.x))
        self.n, self.d = self.x.shape
        self.a = np.random.randn(self.d)
        steps, errors = [], []
        for step in range(self.max_iteration):
            print('\r', step, end='')
            error = (self.Y - self.x @ self.a).T @ (self.Y - self.x @ self.a) / self.n
            steps.append(step)
            errors.append(error)

            grad = self.gradient(self.x, self.Y)
            self.a -= self.learning_rate * grad
            if np.linalg.norm(grad) < self.tolerance_convergence:
                break
        return steps, errors

    def gradient(self, x, Y):
        return 2 * x.T @ (x @ self.a - Y) / self.n

    def predict(self, x):
        if self.intercept:
            x_ = np.hstack((np.ones((x.shape[0], 1)), x))
        else:
            x_ = x
        return x_ @ self.a

    def MSE(self, x, Y):
        return (Y - self.predict(x)).T @ (Y - self.predict(x)) / len(Y)

    def MAE(self, x, Y):
        return abs(Y - self.predict(x)).mean()

    def MAPE(self, x, Y):
        return abs((Y - self.predict(x)) / Y).mean()

model = joblib.load(dump)

columns = ['state', 'size', 'MAE_train', 'MAE_test']
data = np.zeros((0, 4))
counter = 0

for state in range(100):
    for size_tmp in range(1, 100, 1):
        size = size_tmp / 100
        counter = counter + 1
        print(f'\r{counter}. state - {state}; size - {size} ', end='')

        x_train, x_test, Y_train, Y_test = train_test_split(x.copy(), Y.copy(), 
                                                            test_size = size, 
                                                            random_state = state)
        
        row = np.array([state, 
                         size, 
                         model.MAE(x_train, Y_train), 
                         model.MAE(x_test, Y_test)
                         ])
        data = np.vstack((data, row))  


df = pd.DataFrame(data, columns=columns)

print(df.head())

df.to_csv('my-test.csv')
