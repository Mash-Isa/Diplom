import os
import pickle

import joblib
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from ipywidgets import widgets


class DataLoader:

    def choose_file(self, folder):
        file_name_widget = widgets.Dropdown(options=os.listdir(folder),
                                            value=None,
                                            description='File Name')
        ui = widgets.HBox([file_name_widget])

        def print_file(file_name):
            display(file_name)

        out = widgets.interactive_output(print_file, {'file_name': file_name_widget})
        display(ui)
        return out

    def read_data_from_csv(self, folder, selected_file):
        file = selected_file.outputs[0]['data']['text/plain'][1:-1]
        path = f'{folder}/{file}'
        df = pd.read_csv(path)
        df.sort_index(axis=1, inplace=True)
        return self.lower_remove_spaces(df)

    def read_data_from_csv_for_model(self, folder, selected_file):
        file = selected_file.outputs[0]['data']['text/plain'][1:-1]
        path = f'{folder}/{file}'
        df = pd.read_csv(path)
        return df

    def lower_remove_spaces(self, df):
        categorical_columns = list(df.select_dtypes(['object']).columns)
        df[categorical_columns] = df[categorical_columns].astype(str).applymap(str.capitalize).applymap(
            str.strip).replace('Nan', np.nan)
        return df

    def format_columns(self, df, features_to_encode, features_to_drop, have_label=False, label=None, label_name=None):
        df[features_to_encode] = df[features_to_encode].astype(str).applymap(str.lower).applymap(str.strip)
        for i in features_to_encode:
            df = pd.concat([df, pd.get_dummies(df[i], prefix=i)], axis=1)
        df = df.loc[:, ~df.columns.str.endswith('nan')]
        all_features = list(df.select_dtypes(['float', 'uint8']).columns)
        features = list(set(all_features) - set(features_to_drop))
        X = df[features]
        X.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        if have_label:
            y = (df[label] == label_name) * 1
            return X, y
        return X

    def show_df_info(self, df):
        show_df_button = widgets.ToggleButton(description='Show 10 first rows')
        show_nrows_ncols_button = widgets.ToggleButton(description='Show number of rows and columns')
        show_cols_button = widgets.ToggleButton(description='Show columns')
        show_info_button = widgets.ToggleButton(description='Show information')
        show_cols_stats = widgets.ToggleButton(description='Show columns statistics')
        ui = widgets.HBox(
            [show_df_button, show_nrows_ncols_button, show_cols_button, show_info_button, show_cols_stats])

        def buttons_actions(show_df_button=False, show_nrows_ncols_button=False, show_cols_button=False,
                            show_info_button=False, show_cols_stats=False):
            if show_df_button:
                display(df.head(10))
            if show_nrows_ncols_button:
                print(f'Dataset has {df.shape[0]} rows and {df.shape[1]} columns')
            if show_cols_button:
                print('Columns of dataset:')
                print(df.columns.values)
            if show_info_button:
                print(df.info())
            if show_cols_stats:
                display(df.describe())

        out = widgets.interactive_output(buttons_actions,
                                         {'show_df_button': show_df_button,
                                          'show_nrows_ncols_button': show_nrows_ncols_button,
                                          'show_cols_button': show_cols_button,
                                          'show_info_button': show_info_button,
                                          'show_cols_stats': show_cols_stats})
        display(ui, out)


class ModelLauncher:

    def get_model_prediction(self, model_path, X, thr, model_columns_path):
        model = joblib.load(model_path)
        with open(model_columns_path, 'rb') as handle:
            model_columns = pickle.load(handle)
        X = X[model_columns]
        proba = model.predict_proba(X)[:, 1]
        pred = proba > thr
        return proba, pred

    def get_model_decision(self, model_path, model_columns_path, X, thr):
        proba, _ = self.get_model_prediction(model_path=model_path, model_columns_path=model_columns_path, X=X, thr=thr)
        for i in range(X.shape[0]):
            if proba[i] > thr:
                decision = 'Yes'
            else:
                decision = 'No'
            print('_' * 100)
            print()
            print(f'Patient {i + 1}')
            print(f'Model decision for Covid-19: {decision} with Covid-19 probability {round(proba[i], 2)}')
            print('_' * 100)

    def save_predictions_to_file(self, model_path, X, thr, model_columns_path, raw_df, new_file_name):
        proba, pred = self.get_model_prediction(model_path=model_path, model_columns_path=model_columns_path, X=X,
                                                thr=thr)
        predicted_decision = pd.Series(pred).replace({True: 'Yes', False: 'No'})
        raw_df['Model_Probability'] = proba
        raw_df['Model_Decision'] = predicted_decision
        raw_df.to_csv(new_file_name)
        print('Dataset saved!')
