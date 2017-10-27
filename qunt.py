#import Quandl as Quandl
import quandl
import quandl
# mydata = quandl.get("NSE/BAJAJ_AUTO")
# print mydata
import pandas as pd
import quandl as Q
from quandl.errors.quandl_error import NotFoundError

try:
  df = Q.get("GOOG/NYSE_SPY")
  print(df)

except NotFoundError:
  print(NameError)

#GOOG\NYSE_SPY  ,CME/PLZ2016

# data = quandl.get("FRED/GDP")
# #By default, the data is returned as a pandas dataframe.  Quandl can also return data as a numpy array.  To explicitly specify the data download format, do this:
#
# data = quandl.get("FRED/GDP",returns="pandas") 	# pandas data series
# data = quandl.get("FRED/GDP",returns="numpy")		# numpy array
# data = Quandl.get("IMF/POILAPSP_INDEX",frequency="quarterly",startdate="2005",transformation = "normalize",rows="4")
# data.head()
#
# data = quandl.get_table('ZACKS/FC', ticker='AAPL')
# print data GOOG/NYSE_SPY

#quandl.errors.quandl_error.NotFoundError: (Status 404) (Quandl Error QECx02) You have submitted an incorrect Quandl code. Please check your Quandl codes and try again.
#quandl.errors.quandl_error.ForbiddenError: (Status 403) (Quandl Error QEPx04) A valid API key is required to retrieve data. Please check your API key and try again. You can find your API key under your account settings.
