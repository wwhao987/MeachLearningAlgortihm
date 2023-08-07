import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
import missingno as msno
import sys
warnings.filterwarnings('ignore')

# sys.exit()


# def data_info(data):
#     print(data.head())
#     print(data.shape)
#     print(data.info())
#     print(data.describe())
#     print(data.columns)


root = "/Users/afei/PycharmProjects/MLProject/Project_1/telco-churn-prediction-main/telco-churn-prediction-main"
path = os.path.join(root, "telco.csv")
# read data
data = pd.read_csv(path)
data_columns = data.columns
# fig, ax = plt.figure(figsize=(6, 6), dpi=90)

msno.heatmap(data.isnull(), cmap="magma")

plt.show()
