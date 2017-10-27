import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('bmh')

df = pd.read_csv("data/EURUSD60.csv")
df2 = pd.read_csv("data/EURUSD60.csv")

ozone_subset = df.iloc[:, 3,]
ozone_subset2 = df2.iloc[:, 5]

ozone_subset.plot(color="green", linewidth=2, linestyle="-")
ozone_subset2.plot(color="yellow", linewidth=2, linestyle="-")

plt.title('Nitrogen oxides - monthly averages',fontsize = 'large',fontname='Arial',)
plt.ylabel('High',fontname='Arial',fontsize = 'medium')
plt.xlabel('Date',fontname='Arial',fontsize = 'medium')
plt.legend(loc='upper right', fontsize = 'small')
plt.show()