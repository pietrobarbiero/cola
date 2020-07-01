import sys
import scipy
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn import clone
from lazygrid.datasets import load_openml_dataset
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np

from dtl import Fexin, Gexin

datasets = [
    # "iris",
    "seeds",                    # 2417 samples      116 features    2 classes
    "emotions",                        # 2407 samples      299 features    2 classes
    # "isolet",                       # 7797 samples      617 features    26 classes
    # "gina_agnostic",                # 3468 samples      970 features    2 classes
    # "gas-drift",                    # 13910 samples     129 features    6 classes
    # "mozilla4",                     # 15545 samples     6 features      2 classes
    # "letter",                       # 20000 samples     17 samples      26 classes
    # "Amazon_employee_access",       # 32769 samples     10 features     2 classes
    # "electricity",                  # 45312 samples     9 features      2 classes
    # "mnist_784",                    # 70000 samples     785 features    10 classes
    # "covertype",                    # 581012 samples    55 features     7 classes

    # "gisette",                              # 7000 samples      5000 features       2 classes
    # "amazon-commerce-reviews",              # 1500 samples      10000 features      50 classes
    # "OVA_Colon",                            # 1545 samples      10936 features      2 classes
    # "GCM",                                  # 190 samples       16063 features      14 classes
    # "Dexter",                               # 600 samples       20000 features      2 classes
    # "variousCancers_final",                 # 383 samples       54676 features      9 classes
    # "anthracyclineTaxaneChemotherapy",      # 159 samples       61360 features      2 classes
    # "Dorothea",                             # 1150 samples      100000 features     2 classes
]


def main():
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    bar_position = 0
    progress_bar = tqdm(datasets, position=bar_position)
    for dataset in progress_bar:
        progress_bar.set_description("Analysis of dataset: %s" % dataset)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        X, y, n_classes = load_openml_dataset(dataset_name=dataset)
        X = StandardScaler().fit_transform(X)

        N = 10
        model = Gexin(verbose=False)
        model.fit(X, N=N, num_epochs=200, lr=0.01)
        model.compute_sample_graph()
        model.compute_graph()
        model.plot_adjacency_matrix()
        model.plot_graph(y, os.path.join(results_dir, f"{dataset}.png"))
        model.plot_sample_graph(y, os.path.join(results_dir, f"{dataset}_samples.png"))
        # pd.DataFrame(model.adjacency_matrix_).to_csv(os.path.join(results_dir, f"{dataset}.csv"))
        # pd.DataFrame(model.centroids_).to_csv(os.path.join(results_dir, f"{dataset}_centroids.csv"))
        # pd.DataFrame(model.adjacency_samples_).to_csv(os.path.join(results_dir, f"{dataset}_samples.csv"))


if __name__ == "__main__":
    sys.exit(main())
