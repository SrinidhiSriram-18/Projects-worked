import numpy as np
from DecisionTreeRegressor_solu import *
import os
import json


class MyGradientBoostingRegressorSolu():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param learning_rate, type:float
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        int (default=100)
        :param n_estimators, type: integer
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        :param max_depth, type: integer
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node

        estimators: the regression estimators
        '''
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.n_class = None
        self.loss = None
        self.mean = None

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape((1, -1))
            y = y.reshape((-1,))

        self.mean = np.mean(y)
        y_pred = np.empty((X.shape[0],), dtype=np.float64)
        y_pred.fill(self.mean)

        self._fit_stages(X, y, y_pred)

    def predict(self, X):
        y_pred = np.empty((X.shape[0],), dtype=np.float64)
        y_pred.fill(self.mean)
        y_pred = self._predict_stages(X, y_pred)
        return y_pred

    def get_model_dict(self):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = dict()
        for i in range(self.n_estimators):
            model_dict.update({str(i):self.estimators[i].root})

        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

    def _fit_stages(self, X, y, y_pred):
        for m in range(0, self.n_estimators):
            residual = y - y_pred.ravel()
            tree = MyDecisionTreeRegressorSolu(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residual)
            y_pred += self.learning_rate * tree.predict(X).ravel()
            self.estimators[m] = tree

    def _predict_stages(self, X, y_pred):
        for m in range(0, self.n_estimators):
            y_pred += self.learning_rate * self.estimators[m].predict(X).ravel()
        return y_pred


# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            n_estimators = 10 + j * 10
            gbr = MyGradientBoostingRegressor(n_estimators=n_estimators, max_depth=5, min_samples_split=2)
            gbr.fit(x_train, y_train)
            model_dict = gbr.get_model_dict()

            y_pred = gbr.predict(x_train)

            with open("Test_data" + os.sep + "gradient_boosting_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_gradient_boosting_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")
