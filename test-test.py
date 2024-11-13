import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

dump = 'model.dump'
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os, joblib, time

dump = 'model.dump'


# lr-скорость обучения, max_iter-максимальное количество итераций
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

# функция для обучающей модели
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


# импортируем таблицу с датасетом
data = pd.read_csv(r'Video_Games.csv')

# сохраняем в У объём продаж в Японии
Y = data.JP_Sales
Y = np.log(Y+1)
# сохраняем в х датасет без объёма продаж в Японии
x = data.drop(['JP_Sales'], axis = 1)
# сохраняем в х датасет без названий игр
x = x.drop(['Name'], axis = 1)
x = x.drop(['Publisher'], axis = 1)
x = x.drop(['Developer'], axis = 1)
# преобразуем категориальные признаки в числовые
x = pd.concat([x, pd.get_dummies(x[['Platform', 'Genre', 'Rating']], drop_first = True).astype(int)], axis = 1)
# удаляем из датасета столбцы, которые уже преобразовали в числовые признаки
x = x.drop(['Platform', 'Genre', 'Rating'], axis = 1)
# заполняем пропущенные значения NaN указанным числом
x = x.fillna(-10)
# стандартизация
x.Year_of_Release = (x.Year_of_Release - x.Year_of_Release.mean()) / x.Year_of_Release.std()
x.Critic_Score = (x.Critic_Score - x.Critic_Score.mean()) / x.Critic_Score.std()
x.Critic_Count = (x.Critic_Count - x.Critic_Count.mean()) / x.Critic_Count.std()
x.User_Score = (x.User_Score - x.User_Score.mean()) / x.User_Score.std()
x.User_Count = (x.User_Count - x.User_Count.mean()) / x.User_Count.std()
x.NA_Sales = (x.NA_Sales - x.NA_Sales.mean()) / x.NA_Sales.std()
x.EU_Sales = (x.EU_Sales - x.EU_Sales.mean()) / x.EU_Sales.std()
x.Other_Sales = (x.Other_Sales - x.Other_Sales.mean()) / x.Other_Sales.std()

model = joblib.load(dump)

columns = ['state', 'size', 'MAE_train', 'MAE_test']
data = np.zeros((0, 4))
counter = 0

for state in range(1, 1000):
    for size_tmp in range(1, 10, 1):
        size = size_tmp / 10
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
