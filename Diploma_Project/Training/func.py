from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve


class HyperparameterTuner:
    def __init__(self):
        self.strat = StratifiedKFold(n_splits=5,
                                     shuffle=True,
                                     random_state=42)

    def perform_cross_val(self, grid, param, params, X_tr, y_tr, cl_weight):
        train_auc = []
        test_auc = []
        for x in grid:
            params[param] = x
            # print((datetime.now() + timedelta(hours=2)).strftime("%H:%M:%S"), 'Parameters: ')
            # display(params)
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('rf', RandomForestClassifier(**params,
                                                           random_state=42,
                                                           n_jobs=-1,
                                                           class_weight=cl_weight))])
            temp_train_auc = []
            temp_test_auc = []

            for train_index, test_index in self.strat.split(X_tr, y_tr):
                X_train, X_test = X_tr.iloc[train_index], X_tr.iloc[test_index]
                y_train, y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
                pipe.fit(X_train, y_train)
                temp_train_auc.append(roc_auc_score(y_train,
                                                    pipe.predict_proba(X_train)[:, 1]))
                temp_test_auc.append(roc_auc_score(y_test,
                                                   pipe.predict_proba(X_test)[:, 1]))
            train_auc.append(temp_train_auc)
            test_auc.append(temp_test_auc)

            # print('Test CV AUC scores: ')
            # display(temp_test_auc)

        train_auc, test_auc = np.asarray(train_auc), np.asarray(test_auc)

        string = ("Best AUC on CV = {:.3f}\n" +
                  "Achieved at {} = {}")
        grid_max = grid[np.argmax(test_auc.mean(axis=1))]
        print(string.format(max(test_auc.mean(axis=1)),
                            param,
                            grid_max))
        return train_auc, test_auc, grid_max

    def plot_curves(self, grid, train_auc, test_auc, plot_xlabel):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(grid,
                train_auc.mean(axis=1),
                alpha=0.5,
                color='blue',
                label="train")

        ax.plot(grid,
                test_auc.mean(axis=1),
                alpha=0.5,
                color='red',
                label="CV")

        ax.fill_between(grid, test_auc.mean(axis=1) - test_auc.std(axis=1),
                        test_auc.mean(axis=1) + test_auc.std(axis=1),
                        color='#888888',
                        alpha=0.4)

        ax.fill_between(grid, test_auc.mean(axis=1) - 2 * test_auc.std(axis=1),
                        test_auc.mean(axis=1) + 2 * test_auc.std(axis=1),
                        color='#888888',
                        alpha=0.2)

        ax.legend(loc='best')
        ax.set_ylim([0.7, 1.02])
        ax.set_ylabel("AUC")
        ax.set_xlabel(plot_xlabel)

    def perform_tuning(self, params_values_dict: dict(), X_tr, y_tr, cl_weight=None):
        params = {}
        for param, value in params_values_dict.items():
            grid = value
            train_auc, test_auc, grid_best = self.perform_cross_val(grid, param, params, X_tr, y_tr, cl_weight)
            params[param] = grid_best
            self.plot_curves(grid, train_auc, test_auc, param)
        return params


def plot_pr_rec_curves(y_true, y_pred_prob):
    precision, recall, th = precision_recall_curve(y_true.values, y_pred_prob)
    f1 = (2 * precision[1:] * recall[1:]) / (precision[1:] + recall[1:])
    plt.plot(th, precision[1:], 'b', label='Precision')
    plt.plot(th, recall[1:], 'r', label='Recall')
    plt.plot(th, f1, 'g', label='F1')

    plt.title('Recall & Precision & F1 for different threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Recall & Precision & F1')
    plt.legend()
    plt.show()
