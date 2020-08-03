import numpy as np
import os
import json
import operator

class MyDecisionTreeRegressorSolu():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self._build_tree(X, y)

    def predict(self, X):
        predictions = list()
        for row in X:
            prediction = self._predict(self.root, row)
            predictions.append(prediction)
        return np.array(predictions)

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)

    def _build_tree(self, X, y):
        if X.ndim == 1:
            X = X.reshape((1, -1))
            y = y.reshape((-1,))
        dataset = list()
        for i in range(X.shape[0]):
            dataset.append(np.append(X[i], y[i]).tolist())
        self.root = self._get_split(dataset)
        self._split(self.root, 1)

    def _test_split(self, splitting_variable, splitting_threshold, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[splitting_variable] <= splitting_threshold:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def _get_square_error(self, dataset):
        if dataset == []:
            return 0
        error = np.var([row[-1] for row in dataset]) * len(dataset)
        return error

    def _get_split(self, dataset):
        b_splitting_variable, b_splitting_threshold, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
        for splitting_variable in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self._test_split(splitting_variable, row[splitting_variable], dataset)
                error = np.sum(self._get_square_error(group) for group in groups)
                if error < b_score:
                    b_splitting_variable, b_splitting_threshold, b_score, b_groups = splitting_variable, row[splitting_variable], error, groups
        return {'splitting_variable': b_splitting_variable, 'splitting_threshold': b_splitting_threshold, 'groups': b_groups}

    def _to_terminal(self, group):
        outcome = np.mean([row[-1] for row in group])
        return outcome

    def _split(self, node, depth):
        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return
        if len(left) < self.min_samples_split:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_split(left)
            self._split(node['left'], depth + 1)
        if len(right) < self.min_samples_split:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_split(right)
            self._split(node['right'], depth + 1)

    def _predict(self, node, row):
        if row[node['splitting_variable']] <= node['splitting_threshold']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']



def compare_json_dic(json_dic, sample_json_dic):
    if isinstance(json_dic, dict):
        result = 1
        for key in sample_json_dic:
            if key in json_dic:
                result = result * compare_json_dic(json_dic[key], sample_json_dic[key])
                if result == 0:
                    return 0
            else:
                return 0
        return result
    else:
        rel_error = abs(json_dic - sample_json_dic) / np.maximum(1e-8, abs(sample_json_dic))
        if rel_error <= 1e-5:
            return 1
        else:
            return 0


def compare_predict_output(output, sample_output):
    rel_error = (abs(output - sample_output) / np.maximum(1e-8, abs(sample_output))).mean()
    print(rel_error)
    if rel_error <= 1e-5:
        return 1
    else:
        return 0

# For test
if __name__=='__main__':
    for i in range(3):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_dict = tree.get_model_dict()
            y_pred = tree.predict(x_train)

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")



