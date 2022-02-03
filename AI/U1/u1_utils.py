# -*- coding: utf-8 -*-
"""
Author: Schörgenhumer, Brandstetter, Schäfl
Date: 04-10-2021

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement: 
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sys

from distutils.version import LooseVersion
from IPython.core.display import HTML
from sklearn import datasets
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Sequence


def setup_jupyter() -> HTML:
    """
    Setup Jupyter notebook. Warning: this may affect all Jupyter notebooks running on the same Jupyter server.

    :return: HTML instance comprising the modified Jupyter attributes
    """
    return HTML(r"""
    <style>
        .output_png {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
        .jp-RenderedImage {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
    </style>
    <p>Setting up notebook ... finished.</p>
    """)


# noinspection PyUnresolvedReferences
def check_module_versions() -> None:
    """
    Check Python version as well as versions of recommended (partly required) modules.

    :return: None
    """
    python_check = '(\u2713)' if sys.version_info >= (3, 8) else '(\u2717)'
    numpy_check = '(\u2713)' if LooseVersion(np.__version__) >= LooseVersion('1.18') else '(\u2717)'
    pandas_check = '(\u2713)' if LooseVersion(pd.__version__) >= LooseVersion('1.0') else '(\u2717)'
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion('0.23') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')


def load_wine() -> pd.DataFrame:
    """
    Load wine data set [1].

    [1] Forina, M. et al, PARVUS - An Extendible Package for Data Exploration, Classification and Correlation.
        Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa, Italy.

    :return: wine data set
    """
    wine_data = datasets.load_wine()
    data = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
    data['cultivator'] = wine_data['target']
    return data


def load_iris() -> pd.DataFrame:
    """
    Load iris data set [1].

    [1] Fisher,R.A. - The use of multiple measurements in taxonomic problems. Annual Eugenics, 7, Part II, 179-188 (1936)

    :return: iris data set
    """
    iris_data = datasets.load_iris()
    new_col_names = [c.replace(" (cm)", "") for c in iris_data["feature_names"]]
    data = pd.DataFrame(iris_data["data"], columns=new_col_names)
    data["species"] = iris_data["target"]
    return data


def plot_features(data: pd.DataFrame, features: Sequence[str], target_column: Optional[str] = None, **kwargs) -> None:
    """
    Visualizes the specified features of the data set via pairwise relationship plots. Optionally,
    the displayed data points can be colored according to the specified ``target_column``.
    
    :param data: data set containing the features
    :param features: the list of features to visualize
    :param target_column: if specified, color the visualized data points according to this target
    :param kwargs: additional keyword arguments that are passed to ``sns.pairplot``
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(features, Sequence)
    assert target_column is None or isinstance(target_column, str)
    if isinstance(features, str):
        features = [features]
    sns.pairplot(data, vars=features, hue=target_column, palette="deep", **kwargs)


def apply_pca(n_components: int, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Apply principal component analysis (PCA) on specified data set and down-project data accordingly.

    :param n_components: amount of (top) principal components involved in down-projection
    :param data: data set to down-project
    :param target_column: if specified, append target column to resulting, down-projected data set
    :return: down-projected data set
    """
    assert (type(n_components) == int) and (n_components >= 1)
    assert type(data) == pd.DataFrame
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    if target_column is not None:
        projected_data = pd.DataFrame(PCA(n_components=n_components).fit_transform(data.drop(columns=target_column)))
        projected_data[target_column] = data[target_column]
    else:
        projected_data = pd.DataFrame(PCA(n_components=n_components).fit_transform(data))
    return projected_data


def apply_tsne(n_components: int, data: pd.DataFrame, target_column: Optional[str] = None,
               perplexity: float = 10.0) -> pd.DataFrame:
    """
    Apply t-distributed stochastic neighbor embedding (t-SNE) on specified data set and down-project data accordingly.

    :param n_components: dimensionality of the embedding space
    :param data: data set to down-project
    :param target_column: if specified, append target column to resulting, down-projected data set
    :param perplexity: this term is closely related to the number of nearest neighbors to consider
    :return: down-projected data set
    """
    assert (type(n_components) == int) and (n_components >= 1)
    assert type(data) == pd.DataFrame
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    assert (type(perplexity) == float) or (type(perplexity) == int)
    if target_column is not None:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200, init="random").fit_transform(data.drop(columns=target_column)))
        projected_data[target_column] = data[target_column]
    else:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200, init="random").fit_transform(data))
    return projected_data


def apply_k_means(k: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply k-means clustering algorithm on the specified data.

    :param k: amount of clusters
    :param data: data used for clustering
    :return: predicted cluster per data set entry
    """
    assert (type(k) == int) and (k >= 1)
    assert type(data) == pd.DataFrame
    return KMeans(n_clusters=k).fit_predict(data)


def apply_affinity_propagation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply affinity propagation clustering algorithm on the specified data.

    :param data: data used for clustering
    :return: predicted cluster per data set entry
    """
    assert type(data) == pd.DataFrame
    return AffinityPropagation(affinity='euclidean', random_state=None).fit_predict(data)


def plot_points_2d(data: pd.DataFrame, target_column: Optional[str] = None, legend: bool = True, **kwargs) -> None:
    """
    Visualize data points in a two-dimensional plot, optionally color-coding according to ``target_column``.

    :param data: data set to visualize
    :param target_column: optional target column to be used for color-coding
    :param legend: flag for displaying a legend
    :param kwargs: optional keyword arguments passed to ``plt.subplots``
    :return: None
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] in [2, 3])
    assert (target_column is None) or ((data.shape[1] == 3) and (data.columns[2] == target_column))
    assert type(legend) == bool
    fig, ax = plt.subplots(**kwargs)
    color_targets = data[target_column] if target_column is not None else None
    color_palette = sns.color_palette()[:len(set(color_targets))] if color_targets is not None else None
    try:
        _ = sns.scatterplot(x=data[0], y=data[1], hue=color_targets, ax=ax, palette=color_palette)
    except ValueError:
        _ = sns.scatterplot(x=data[0], y=data[1], hue=color_targets, ax=ax, legend=r'full')
    plt.tight_layout()
    plt.show()
