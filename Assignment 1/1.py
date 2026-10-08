from pandas.core.ops.docstrings import key
from sqlalchemy.sql.elements import NamedColumn
from pandas.core.interchange import column
from email import header
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

print ("Succ")

Data = pd.read_excel("Assignment 1/VARSWE.xls",index_col=0)

print(Data.dtypes)

for i in Data.columns:
    pd.to_numeric(Data[i])

Data_punkter = {"GDP":Data["GDP"], "Annual inflation rate":Data["Annual inflation rate"], "CPI seasonal adjusted":Data["CPI seasonal adjusted"],"Domestic interest rate":Data["Domestic interest rate"], "Oil price":Data["Oil price"],"Real effective exchange rate":Data["Real effective exchange rate"],"Foreign interest rate": Data["Foreign interest rate"],"Federal Funds rate": Data["Federal Funds rate"]}

Data_punkter_navne = list(Data_punkter.keys())

print(Data_punkter)

Data_punkter["Annual inflation rate"].index.shape

for i in Data_punkter_navne:
    if Data_punkter[i].any() > 0 :
        print("no zero",Data_punkter[i].head())
    elif Data_punkter[i].any() == 0:
        print("has 0",Data_punkter[i])

for i in Data_punkter_navne:
    print(i,Data_punkter[i].isna().sum())


for i in Data_punkter_navne:
    Data_punkter[i] = np.log(Data_punkter[i])

Data_punkter = pd.DataFrame(Data_punkter)

Data_punkter = Data_punkter.dropna()

Data_punkter_navne_uden_FFR = Data_punkter_navne
Data_punkter_navne_uden_FFR.remove(list[''])

print(Data_punkter_navne_uden_FFR)


for i in Data_punkter_navne_uden_FFR:
    Data_punkter_first_diff[i] = Data_punkter[i].pct_change()

Data_punkter_first_diff = Data_punkter.pct_change()
Data_punkter_first_diff = Data_punkter_first_diff.dropna()


plt.close()
plt.clf()
plt.figure(figsize=(12,8))
for i in Data_punkter_navne:
    plt.plot(Data_punkter_first_diff["Annual inflation rate"].index,Data_punkter_first_diff[i])
plt.legend(Data_punkter_first_diff.keys())
plt.grid()
plt.show()

