

import IPython.core.display as di
# This line will hide code by default when the notebook is exported as HTML
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
from IPython.core.display import HTML
from __future__ import division

import pandas as pd 
import numpy as np
from dateutil import rrule, parser

from utilplot import pointplot, scatterplot, acf, pacf
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot



#readin 
data = pd.read_csv('Case Study Raw Data .csv', sep = ',')

#create a datetime index
date1 = '2014-12-14'
date2 = '2015-02-10'
dates = list(rrule.rrule(rrule.DAILY,
                         dtstart=parser.parse(date1),
                         until=parser.parse(date2)))

#pivot table to a long table 
data_reshape = pd.DataFrame(data.set_index(['date']).stack().reset_index())
data_reshape.rename(columns={'date': 'date', 'level_1': 'phone', 0 : 'revenue'}, inplace=True)


#plot time series 
pointplot(data_reshape, 'date', 'revenue', 'phone', 
          'Revenue Over Time', 'Date', 'Revenue', 
           'muted', (60, 20))


#copy raw data for modeling 
data_for_model = data.copy(deep = True)
data_for_model.index = dates

#relationship between iphone and android 
scatterplot(data_for_model.loc['2014-12-14':'2015-01-31'], 
            'android revenue', 'iphone revenue', #x, y variable 
            'Scatterplot of Android and Iphone revenue', 'Android revenue','Iphone revenue') #labels 


#fit a linear regression and predict android revenue after 02-01 if there is no product change 
def OLS_pred(y_train, X_train, X_test, table, pred):
    """Fit a linear regression and predict value after launch.
       Create a long table with actual and predicted revenue.
       Args: y_train
             X_train
             X_test
             table: table contains actual revenues for android and iphone 
             pred: new column name to store predicted value 
      
    """
    #fit OLS and predict 
    android_iphone_OLS = sm.regression.linear_model.OLS(y_train, X_train).fit()
    yhat = android_iphone_OLS.predict(X_test)
    #add one column for prediction 
    table_new = table.copy(deep = True)
    table_new[pred] = list(y_train)+list(yhat)
    #pivot table to a long table 
    data_for_model_reshape = pd.DataFrame(table_new.set_index(['date']).stack().reset_index())
    data_for_model_reshape.rename(columns={'date': 'date', 'level_1': 'phone', 0 : 'revenue'}, inplace=True)
    return data_for_model_reshape, yhat 


#predict android revenue 
y_train = data_for_model.loc['2014-12-14':'2015-01-31']['android revenue']
X_train = data_for_model.loc['2014-12-14':'2015-01-31']['iphone revenue']
X_test = data_for_model.loc['2015-02-01':'2015-02-10']['iphone revenue']
android_pred_table, android_yhat = OLS_pred(y_train, X_train, X_test, data_for_model,'android revenue if no intervention')

#visualization 
pointplot(android_pred_table, 'date', 'revenue', 'phone', 
          'Revenue Over Time', 'Date', 'Revenue', 
           'muted', (60, 20))


#predict iphone revenue 
X_test = data_for_model.loc['2015-02-01':'2015-02-10']['android revenue']
iphone_pred_table, iphone_yhat = OLS_pred(X_train, y_train, X_test, data_for_model,'iphone revenue if launch')

#visualization 
pointplot(iphone_pred_table, 'date', 'revenue', 'phone', 
          'Revenue Over Time', 'Date', 'Revenue', 
           'muted', (60, 20))

#calculate incremental revenue 
result = pd.DataFrame()
result['android incremental revenue'] = list(data_for_model.loc['2015-02-01':'2015-02-10']['android revenue']) - android_yhat 
result['iphone incremental revenue'] = iphone_yhat-list(data_for_model.loc['2015-02-01':'2015-02-10']['iphone revenue']) 
result.index = data_for_model.index[-10:]
result


####method2 ARMA model 
#seperate data by the launch date 
data_prior = data_for_model.loc['2014-12-14':'2015-01-31']
data_post = data_for_model.loc['2015-02-01':'2015-02-10']

#acf and pacf plots 
acf(data_prior['android revenue'].values.squeeze(), 20, 'Autocrrelation Android', 'Lag', 'Autocorrelation', (30,5))
pacf(data_prior['android revenue'], 20, 'Partial Autocrrelation Android', 'Lag', 'Partial Autocorrelation', (30,5))


#fit model and predict revenue after launch date
ARMA_android = sm.tsa.ARMA(data_prior['android revenue'], (1, 2), exog = data_prior['iphone revenue']).fit()
android_pred = ARMA_android.predict('2015-02-01', '2015-02-10', exog = data_post['iphone revenue'], dynamic=False)

#check residual 
acf(ARMA_android.resid.values.squeeze(), 20, 'Autocrrelation of residual Android', 'Lag', 'Autocorrelation of residual', (30, 5))
pacf(ARMA_android.resid, 20, 'Partial Autocorrelation of residual Android', 'Lag', 'Partial Autocorrelation of residual', (30, 5))


#check Iphone acf and pacf 
acf(data_prior['iphone revenue'].values.squeeze(), 20, 'Autocrrelation Iphone', 'Lag', 'Autocorrelation', (30,5))
pacf(data_prior['iphone revenue'], 20, 'Partial Autocrrelation Iphone', 'Lag', 'Partial Autocorrelation', (30,5))

#fit iphone with ARMA model 
ARMA_iphone = sm.tsa.ARMA(data_prior['iphone revenue'], (1, 2), exog = data_prior['android revenue']).fit()
iphone_pred = ARMA_iphone.predict('2015-02-01', '2015-02-10', exog = data_post['android revenue'], dynamic=False)

#check residual 
acf(ARMA_iphone.resid.values.squeeze(), 20, 'Autocrrelation of residual Iphone', 'Lag', 'Autocorrelation of residual', (30, 5))
pacf(ARMA_iphone.resid, 20, 'Partial Autocorrelation of residual Iphone', 'Lag', 'Partial Autocorrelation of residual', (30, 5))


#output the incremental revenue 
result = pd.DataFrame()
result['android incremental revenue'] =np.subtract(list(data_for_model.loc['2015-02-01':'2015-02-10']['android revenue']), list(android_pred))
result['iphone incremental revenue'] = np.subtract(list(iphone_pred), list(data_for_model.loc['2015-02-01':'2015-02-10']['iphone revenue']))
result.index = data_for_model.index[-10:]
result

