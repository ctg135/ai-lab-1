import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os, joblib, time

dump = 'model.dump'

# Импорт датасета
data = pd.read_csv(r'Video_Games.csv')

# Сохранение в У объёма продаж в Японии
Y = data.JP_Sales
Y = Y/10

# Сохранение датасета без объёма продаж в Японии
x = data.drop(['JP_Sales'], axis = 1)

# Удаление названий игр
x = x.drop(['Name'], axis = 1)

# Список категориальных признаков
categorical_columns = x.select_dtypes(include=['object']).columns.tolist()

# Создание экземпляра энкодера
encoder = OneHotEncoder(sparse_output=False)

# Полученние значений после one hot encoding
one_hot_encoded = encoder.fit_transform(x[categorical_columns])

# Преобразование полученных значений в массив типа int
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns), dtype=int)   

# Конкатенаця полученных значений в результате one hot encoding
x = pd.concat([x, one_hot_df], axis=1)

# Удаляем оставшиеся категориальные признаки
x = x.drop(categorical_columns, axis=1)


# Заполняем пропущенные значения NaN указанным числом
x = x.fillna(10)

# Стандартизация числовых полей
for col in ['Year_of_Release', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count',
            'NA_Sales', 'EU_Sales', 'Other_Sales']:
    x[col] = (x[col] - x[col].mean()) / x[col].std()

# Функция микширования датасета в данные обучения и тестирования
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

# Смешиваем данные для обучения и для тестов
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.6, random_state = 1)

# Перевод переменных в массив numpy
x_train = x_train.to_numpy()
Y_train = Y_train.to_numpy()
x_test = x_test.to_numpy()
Y_test = Y_test.to_numpy()

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

# Проверка на существование модели
if os.path.exists(dump):
    # Выгрузка сохраненной модели
    print('Модель уже обучена, выгружаем')
    model = joblib.load(dump)

else:
    # Создание и обучение модели
    start_time = time.time()
    print('Обучаем модель')

    model = LinearRegressionGD(lr = 0.1, max_iter = 200000, tol = 1e-6, intercept = True)
    steps, errors = model.fit(x_train, Y_train)

    print('\nВремя обучения: ', time.time() - start_time)

    # Сохранение дампа объекта модели
    joblib.dump(model, dump)


# Вывод расчетных данных
print('Коэффициенты: ', model.a)
print('MSE train / test', model.MSE(x_train, Y_train), ' / ', model.MSE(x_test, Y_test))
print('MAE train / test', model.MAE(x_train, Y_train), ' / ', model.MAE(x_test, Y_test))

