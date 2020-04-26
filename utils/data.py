import pandas as pd


def load_data():
    df = pd.read_csv('data/iris_data.csv')
    features = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
    labels = df['Class']
    return features, labels
