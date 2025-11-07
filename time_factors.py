import pandas as pd


file='/projects/IFDM/Flomove/Processing/2023/Results_vlabru_withLEZ/Timeprofile.csv'

time_factors=pd.read_csv(file)

print(time_factors.head())

factors_hour=time_factors[['hour','0','1','2','3','4','5']].groupby('hour').mean()

factors_hour.mean(axis=1).to_csv('time_factors_hourly_mean.csv')
