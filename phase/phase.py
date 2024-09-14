import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

Flat = [
    3.676285714285695,
    3.1753333333333407,
    3.4382857142857284,
    3.426047619047592,
    3.5849374999999384
]

Terrain = [
    4.592066666666657,
    4.011777777777764,
    3.664812500000025,
    3.4173846153846887,
    3.9766470588235063
]

amputate_L3 = [
    4.7486363636363755,
    3.7619230769230922,
    4.884538461538446,
    3.749000000000056,
    4.186749999999961
]

amputate_r2 = [
    6.627000000000019,
    6.65711111111111,
    5.087695652173926,
    4.878999999999981,
    5.929421052631537
]

amputate_r3_l3 = [
    3.5844999999999914,
    4.6632857142857125,
    3.646159999999999,
    4.13357142857143,
    4.595789473684212
]

amputate_r2_l2 = [
    5.728875000000016,
    9.046833333333325,
    6.931642857142849,
    11.675900000000047,
    9.107785714285715
]

data={'Flat':Flat,'Terrain':Terrain,'amputate_L3':amputate_L3,'amputate_r2':amputate_r2,'amputate_r3_l3':amputate_r3_l3,'amputate_r2_l2':amputate_r2_l2}

def calculate_statistics():
    saving = []
    keys = list(data.keys())
    flat = data.get('Flat')
    for key in keys:
        if key != 'Flat':
            terrain = data.get(key)
            stat, p = mannwhitneyu(flat, terrain, alternative='less')
            saving.append([key, stat, p])
    return saving

saving = calculate_statistics()
df = pd.DataFrame(saving, columns=['Perturbation', 'Stat', 'p-value'])
df.to_csv('phase.csv', index=False)

            