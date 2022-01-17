import random

import numpy as np
import imageio
import glob

from main import letter_to_y
from network import Dense, Network

import pandas as pd
import plotly.express as px


def load_aadt():
    df = pd.read_csv('cars_regression/aadt.csv')
    df = df.drop_duplicates(keep=False, subset='sid')
    df_x = df[['avrs', 'frc', 'effectiveLanes']]
    df_y = df[['aadt']]
    # fig = px.scatter(df, x='avrs', y='aadt', color='frc', hover_data=df, title='Aadt regression')
    # fig.show()

    return df_x.to_numpy(), df_y.to_numpy()


def test_aadt_regression_single_layer():
    xs, ys = load_aadt()

    speed = 0.000000000001

    network = Network(epochs=50, layers=[
        Dense(in_shape=3, out_shape=1, learn_speed=speed),
    ])

    network.teach(xs, ys)
    network.plot_loss(from_epoch=5)
    network.plot_mpes(from_epoch=0)
    network.plot_mae(from_epoch=0)


def test_aadt_regression_multy_layer():
    xs, ys = load_aadt()

    speed = 0.00000000000000000001

    network = Network(epochs=10000, layers=[
        Dense(in_shape=3, out_shape=5, learn_speed=speed),
        Dense(in_shape=5, out_shape=1, learn_speed=speed)
    ])

    network.teach(xs, ys)
    network.plot_loss(from_epoch=5)
    network.plot_mpes(from_epoch=0)
    network.plot_mae(from_epoch=0)
