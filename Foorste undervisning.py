import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("succ")

PATH = "VARSWE-1.xls"


# Opgave 1
df = pd.read_excel(PATH, index_col=0,header=4)
df = df.iloc[1:, :].reset_index(drop=True)

date = pd.read_excel(PATH,header=5)

df["Dates"] = date.iloc[:,0]
#this is just to have the quarterly date structure
df["Dates"] = pd.to_datetime(df["Dates"], format="%b-%y")
#we make a new column with quarter dates
df["Quarter"] = df["Dates"].dt.to_period("Q")

#now we use the quarters as the index for the datatime.
#makes time series analysis and plotting a bit easier.
df = df.set_index("Quarter").sort_index()

df = df.drop(columns=["Dates"])  # we re-create a clean Dates at the end


# Let's make the variables ready! :)
# Here, I make use of numpy, but you can also just do it directly in pandas...
# =============================================================================

# GDP in logs

print(df.dtypes)

df = pd.to_numeric(df["GDP"])

GDP_log = np.log(df["GDP"])

print(df.columns)
print(df.head())

# This is the canadian domestic int. rate (this is the 'r' column in the new file)
IntRate = df["Domestic interest rate"]

# Fed funds rate from 'FED': this means "foreign" us interest rate
FedFunds = df["Federal Funds rate"]


# CPI inflation:
# We want year on year inflammation, and thus, we take the 4-shift=1 year.
#multiply to get in percentages
Infl_yoy = df["Annual inflation rate"].pct_change()
Infl = Infl_yoy.rename("Infl")

# We calc the exchange rate change as a quarter-over-quarter log change from RERinv variable
# also get in percent
ExchangeRate = df["Real effective exchange rate"].pct_change()
ExchangeRate = ExchangeRate.rename("ExchangeRate")

# --------------------------
# We make the final dataframe ready!
# --------------------------
result = pd.concat([FedFunds, GDP_log, Infl, IntRate, ExchangeRate], axis=1)
result = result.dropna()  # drop first few rows lost to differences.
# Put a friendly Dates column back (quarter start timestamps)
result = result.reset_index().rename(columns={"Quarter": "Dates"})
result["Dates"] = result["Dates"].dt.to_timestamp(how="start")

# Reorder as you wish
df = result[["Dates", "FedFunds", "GDP", "Infl", "IntRate", "ExchangeRate"]]

#For cleaning
del ExchangeRate, GDP_log, Infl, Infl_yoy, IntRate, PATH, result, FedFunds

#Show the data
df

