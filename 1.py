import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("succ")

Data = pd.read_excel("/Users/nikolaizwisler/Downloads/EUR.xls",index_col=0,header=0)



D_Euro = Data["EURO"].pct_change()*100

plt.close()
plt.clf()
plt.figure(figsize=(12,8))
plt.plot(D_Euro, color="red")
plt.grid()
plt.ylabel("Log udvikling")
plt.xlabel("tid")
plt.legend(['Udvikling'])
plt.show()

print("Her har vi maksimum og minimum i pct ændring", np.max(Data), np.min(Data))
print("Her har vi maksimum og minimum i pct ændring",np.max(D_Euro), np.min(D_Euro))