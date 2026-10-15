from matplotlib.pyplot import figure
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

data_navne = Data_punkter.keys()

print(Data)

Data_punkter["GDP_log"] = np.log(Data_punkter["GDP"])

Data_punkter["REER_log_diff"] = np.log(Data_punkter["Real effective exchange rate"]).diff(1)

plt.close()
plt.clf()

fig, ax = plt.subplots(3, 2, figsize=(12, 8))

ax = ax.flatten()

y_labels = {
    "GDP": "GDP (millions)",
    "Annual inflation rate": "Inflation rate",
    "CPI seasonal adjusted": "Index",
    "Domestic interest rate": "Percent (%)",
    "Oil price": "Price",
    "Real effective exchange rate": "Exchange rate",
    "Foreign interest rate": "Percent (%)",
    "Federal Funds rate": "Percent (%)",
    "GDP_log": "log values",
    "REER_log_diff": "log Diff"
}

data_navne_Question_1 = Data_punkter.copy()


for i in ["GDP","CPI seasonal adjusted","Oil price", "Real effective exchange rate", "Federal Funds rate"]:
    data_navne_Question_1.pop(i, None)


for i, navn in enumerate(data_navne_Question_1):
    ax[i].plot(
        Data_punkter[navn],
        label=navn
        
    )

    ax[i].set_ylabel(y_labels[navn])
    ax[i].set_title(navn)
    ax[i].legend(loc="upper right")
    ax[i].grid(True)

plt.tight_layout()
plt.show()

data_navne_question_1 = data_navne_Question_1.keys()
for i in data_navne_question_1:
    print(Data_punkter[i].describe())
