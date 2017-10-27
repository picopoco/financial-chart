import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
headers = ['Date','Time','Open','High','Low','close','Volume']

df = pd.read_csv('data/EURUSD60.csv',parse_dates=     {"Datetime" : [1,2]},names=headers)

print (df)

df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y.%m.%d'))

#df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y.%m.%d'))
x = df['Date']
y1 = df.iloc[:, 2 ]
y2 = df.iloc[:, 2]


plt.plot(color="green", linewidth=2, linestyle="-")
plt.plot(color="azure", linewidth=2, linestyle="-")


plt.title('Nitrogen oxides - monthly averages',fontsize = 'large',fontname='Arial',)
plt.ylabel('NOx ugm-3',fontname='Arial',fontsize = 'medium')
plt.xlabel('Month number',fontname='Arial',fontsize = 'medium')
plt.legend(loc='upper right', fontsize = 'small')
plt.show()

#y = df[:,3]

# # plot
# plt.plot(x,y1)
# # beautify the x-labels
# plt.gcf().autofmt_xdate()
#
# plt.show()