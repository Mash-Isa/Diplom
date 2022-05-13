from IPython.core.display_functions import display
import ipywidgets as widgets
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


class Plotter:

    def plot_scatter(self, df):
        numerical_columns = list(df.select_dtypes(['float']).columns)
        categorical_columns = list(df.select_dtypes(['object']).columns)
        x = widgets.Dropdown(
            options=numerical_columns,
            value=numerical_columns[0],
            description='X:')
        y = widgets.Dropdown(
            options=numerical_columns,
            value=numerical_columns[1],
            description='Y:')
        color = widgets.Dropdown(
            options=categorical_columns,
            value=categorical_columns[0],
            description='Color:')
        log_x = widgets.Checkbox(description='log_x', value=False)
        log_y = widgets.Checkbox(description='log_y', value=False)
        plot_button = widgets.ToggleButton(description='Plot Scatter')
        ui = widgets.VBox([widgets.HBox([x, log_x, plot_button]), widgets.HBox([y, log_y]), widgets.HBox([color])])

        def plot_features(x=numerical_columns, y=numerical_columns, color=categorical_columns, log_x=True, log_y=True,
                          plot_button=False):
            if plot_button:
                fig = px.scatter(df, x=x, y=y, color=color, log_x=log_x, log_y=log_y, size_max=20)
                fig.show()

        out = widgets.interactive_output(plot_features,
                                         {'x': x, 'y': y, 'color': color, 'log_x': log_x, 'log_y': log_y,
                                          'plot_button': plot_button})
        display(ui, out)

    def kdeplot_single(self, ax, df, is_0, feature, log_x, disease_name):
        # prepare data for plotting
        k_band_width = 25
        kde_data = np.log10(df[df[feature] > 0][feature]) if log_x == True else df[df[feature] >= 0][feature]
        data = kde_data.values
        if len(data) > 0:
            kde = KernelDensity(kernel='gaussian', bandwidth=(data.max() or 0.1) /
                                                             k_band_width).fit(data.reshape(-1, 1))
            _linspace = np.linspace(data.min(), data.max(), 100)
            kde_v = np.exp(kde.score_samples(_linspace.reshape(-1, 1)))
            gr_extr = argrelextrema(kde_v, np.greater)[0]
            less_extr = argrelextrema(kde_v, np.less)[0]
            # plotting
            label = f'not-{disease_name}' if is_0 else disease_name
            label = label + ': ' + str(len(data))
            ax.plot(_linspace, kde_v, label=label)
            ax.scatter(_linspace[gr_extr], kde_v[gr_extr], c='#aa0000')
            ax.scatter(_linspace[less_extr], kde_v[less_extr], c='#00aa00')
            ax.legend(fontsize=10)
            ax.set_xlabel(feature)
            ax.grid(True)
        else:
            print('No data')

    def plot_distribution(self, df, column, disease_name):
        df_1 = df[df[column] == disease_name]
        df_0 = df[df[column] != disease_name]
        numerical_columns = list(df.select_dtypes(['float']).columns)
        x = widgets.Dropdown(
            options=numerical_columns,
            value=numerical_columns[0],
            description='X:')
        log_x = widgets.Checkbox(description='log_x', value=False)
        plot_button = widgets.ToggleButton(description='Plot Distribution')
        ui = widgets.HBox([x, log_x, plot_button])

        def plot_features(x=numerical_columns, log_x=True, plot_button=False):
            if plot_button:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 4))
                self.kdeplot_single(ax, df_1, is_0=0, feature=x, log_x=log_x, disease_name=disease_name)
                self.kdeplot_single(ax, df_0, is_0=1, feature=x, log_x=log_x, disease_name=disease_name)
                ax.set_title(f'Distribution of {x}')
                fig.show()

        out = widgets.interactive_output(plot_features,
                                         {'x': x, 'log_x': log_x, 'plot_button': plot_button})
        display(ui, out)

    def plot_bar_chart(self, df):
        categorical_columns = list(df.select_dtypes(['object']).columns)
        x = widgets.Dropdown(
            options=categorical_columns,
            value=categorical_columns[0],
            description='X:')
        splitting = widgets.Dropdown(
            options=categorical_columns,
            value=categorical_columns[1],
            description='Splitting:')
        plot_button = widgets.ToggleButton(description='Plot Bar')
        ui = widgets.VBox([widgets.HBox([x, plot_button]), widgets.HBox([splitting])])

        def plot_features(x=categorical_columns, splitting=categorical_columns, plot_button=False):
            if plot_button:
                sns.countplot(x=x, hue=splitting, data=df)

        out = widgets.interactive_output(plot_features,
                                         {'x': x, 'splitting': splitting, 'plot_button': plot_button})
        display(ui, out)

    def plot_distplot(self, df):
        numerical_columns = list(df.select_dtypes(['float']).columns)
        x = widgets.Dropdown(
            options=numerical_columns,
            value=numerical_columns[0],
            description='X:')
        plot_button = widgets.ToggleButton(description='Plot Distplot')
        ui = widgets.VBox([widgets.HBox([x, plot_button])])

        def plot_features(x=numerical_columns, plot_button=False):
            if plot_button:
                sns.distplot(df[x])

        out = widgets.interactive_output(plot_features,
                                         {'x': x, 'plot_button': plot_button})
        display(ui, out)

    def plot_boxplot(self, df):
        numerical_columns = list(df.select_dtypes(['float']).columns)
        x = widgets.Dropdown(
            options=numerical_columns,
            value=numerical_columns[0],
            description='X:')
        plot_button = widgets.ToggleButton(description='Plot Boxplot')
        ui = widgets.VBox([widgets.HBox([x, plot_button])])

        def plot_features(x=numerical_columns, plot_button=False):
            if plot_button:
                _, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
                sns.boxplot(data=df[x], ax=axes[0])
                sns.violinplot(data=df[x], ax=axes[1])

        out = widgets.interactive_output(plot_features,
                                         {'x': x, 'plot_button': plot_button})
        display(ui, out)

    def plot_corr(self, df):
        numerical_columns = list(df.select_dtypes(['float']).columns)
        plot_button = widgets.ToggleButton(description='Plot Correlation')
        ui = widgets.VBox([widgets.HBox([plot_button])])

        def plot_features(plot_button=False):
            if plot_button:
                corr_matrix = df[numerical_columns].corr()
                sns.heatmap(corr_matrix)

        out = widgets.interactive_output(plot_features,
                                         {'plot_button': plot_button})
        display(ui, out)

    def plot_hist(self, df):
        numerical_columns = list(df.select_dtypes(['float']).columns)
        plot_button = widgets.ToggleButton(description='Plot Histograms')
        ui = widgets.VBox([widgets.HBox([plot_button])])

        def plot_features(plot_button=False):
            if plot_button:
                df[numerical_columns].hist(figsize=(20, 20))

        out = widgets.interactive_output(plot_features,
                                         {'plot_button': plot_button})
        display(ui, out)
