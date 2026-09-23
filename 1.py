import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("succ")

Data = pd.read_excel("/Users/nikolaizwisler/Downloads/EUR.xls",index_col=0,header=0)



D_Euro = Data["EURO"].pct_change()

plt.clf()
plt.plot(D_Euro)
plt.grid()
plt.ylabel("Log udvikling")
plt.xlabel("tid")
plt.show()