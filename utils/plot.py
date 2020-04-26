import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.tree import export_graphviz

FIGURES_DIR = 'figures/'
GRAPH_DIR = 'graph/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def plot_cm(cm, name):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.title(name)
    plt.savefig(FIGURES_DIR + f'Figure_{name}' + '.png')
    plt.show()


def graphviz(model,feature_names, label_names,graph_name):
    dot_path = GRAPH_DIR + graph_name +'.dot'
    png_path = FIGURES_DIR + graph_name + '.png'
    export_graphviz(
        model,
        out_file=dot_path,
        feature_names=feature_names,
        class_names=label_names,
        rounded=True,
        filled=True)

    os.system(f'dot -Tpng {dot_path} -o {png_path}')

