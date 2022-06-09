#!/usr/bin/env python
# coding: utf-8

# # def final_fun_1(X):

# In[1]:



import seaborn as sns
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


def final_fun_1(PdM_telemetry, PdM_errors, PdM_maint, PdM_machines, finalized_model):
    '''
    This function returns status of failure of machines' components with the following inputs:
    
    PdM_telemetry = Hourly average data of voltage, rotation, pressure, vibration 
                    collected from machines. 
   
    PdM_errors = Errors encountered by the machines in operating condition.
    PdM_maint = Replacement of component history.
    PdM_machines = Model type & age of the Machines (Metadata of machine).
    finalized_model = A pre-trained Machine Learning model. 
    
    '''

    #Loading all the datasets using Pandas library
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    telemetry = pd.read_csv(PdM_telemetry, nrows= 24) #66666
    errors = pd.read_csv(PdM_errors)
    maint = pd.read_csv(PdM_maint)
    machines = pd.read_csv(PdM_machines)

    # Formating datetime field.
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
    errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
    errors['errorID'] = errors['errorID'].astype('category')
    maint['datetime'] = pd.to_datetime(maint['datetime'], format="%Y-%m-%d %H:%M:%S")
    maint['comp'] = maint['comp'].astype('category')
    machines['model'] = machines['model'].astype('category')

    #Lag Features from Telemetry data
    # Calculate "resample min values" over the last 3 hour lag window for telemetry features.
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                            values=col).resample('3H', closed='left', label='right').min().unstack())

    telemetry_min_3h = pd.concat(temp, axis=1)
    telemetry_min_3h.columns = [i + '_min_3h' for i in fields]
    telemetry_min_3h.reset_index(inplace=True)

    # Calculate "resample max values" over the last 3 hour lag window for telemetry features.
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                            values=col).resample('3H', closed='left', label='right').max().unstack())

    telemetry_max_3h = pd.concat(temp, axis=1)
    telemetry_max_3h.columns = [i + '_max_3h' for i in fields]
    telemetry_max_3h.reset_index(inplace=True)

    # Calculate "resample mean values" over the last 3 hour lag window for telemetry features.
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                            values=col).resample('3H', closed='left', label='right').mean().unstack())

    telemetry_mean_3h = pd.concat(temp, axis=1)
    telemetry_mean_3h.columns = [i + '_mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace=True)

    # Calculate "resample standard deviation" over the last 3 hour lag window for telemetry features.
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                            values=col).resample('3H', closed='left', label='right').std().unstack())

    telemetry_sd_3h = pd.concat(temp, axis=1)
    telemetry_sd_3h.columns = [i + '_sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace=True)

    #Capturing a longer term effect, 24 hour lag features

    #Calculate "rolling min" over the last 24 hour lag window for telemetry features.

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
            temp.append(pd.pivot_table(telemetry,  index='datetime',
                                                   columns='machineID',
                                        values=col).rolling(window=24,center=False).min().resample('3H',
                                                                        closed='left', 
                                                                    label='right').first().unstack())                                                                                


    telemetry_min_24h = pd.concat(temp, axis=1)       
    telemetry_min_24h.columns = [i + '_min_24h' for i in fields]
    telemetry_min_24h.reset_index(inplace=True)
    telemetry_min_24h = telemetry_min_24h.loc[-telemetry_min_24h['volt_min_24h'].isnull()]

    #Calculate "rolling max" over the last 24 hour lag window for telemetry features.

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
            temp.append(pd.pivot_table(telemetry,  index='datetime',
                                                   columns='machineID',
                                    values=col).rolling(window=24,center=False).max().resample('3H', 
                                                                                        closed='left', 
                                                                        label='right').first().unstack())                                                                                


    telemetry_max_24h = pd.concat(temp, axis=1)       
    telemetry_max_24h.columns = [i + '_max_24h' for i in fields]
    telemetry_max_24h.reset_index(inplace=True)
    telemetry_max_24h = telemetry_max_24h.loc[-telemetry_max_24h['volt_max_24h'].isnull()]

    #Calculate "rolling mean" over the last 24 hour lag window for telemetry features.

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
            temp.append(pd.pivot_table(telemetry,  index='datetime',
                                                   columns='machineID',
                                        values=col).rolling(window=24,center=False).mean().resample('3H',
                                                                                    closed='left', 
                                                                label='right').first().unstack())                                                                                


    telemetry_mean_24h = pd.concat(temp, axis=1)       
    telemetry_mean_24h.columns = [i + '_mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace=True)
    telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['volt_mean_24h'].isnull()]

    #Calculate "rolling standard deviation" over the last 24 hour lag window for telemetry features.

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
            temp.append(pd.pivot_table(telemetry,  index='datetime',
                                                   columns='machineID',
                                values=col).rolling(window=24,center=False).std().resample('3H', 
                                                    closed='left', label='right').first().unstack())   

    telemetry_sd_24h = pd.concat(temp, axis=1)
    telemetry_sd_24h.columns = [i + '_sd_24h' for i in fields]
    telemetry_sd_24h.reset_index(inplace=True)
    telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['volt_sd_24h'].isnull()]

    # Merge columns of feature sets created earlier
    telemetry_feat = pd.concat([telemetry_min_3h,
                                telemetry_max_3h.iloc[:, 2:6],
                                telemetry_mean_3h.iloc[:, 2:6],
                                telemetry_sd_3h.iloc[:, 2:6],
                                telemetry_min_24h.iloc[:, 2:6],
                                telemetry_max_24h.iloc[:, 2:6],
                                telemetry_mean_24h.iloc[:, 2:6],
                                telemetry_sd_24h.iloc[:, 2:6]], axis=1).dropna()


    #Lag Features from Errors dataset
    # Create a column for each error type
    error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
    error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
    # Combine errors for a given machine in a given hour
    error_count = error_count.groupby(['machineID', 'datetime']).sum().reset_index()
    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], 
                                                             how='left').fillna(0.0)

    #Total number of errors of each type over the last 24 hours
    temp = []
    fields = ['error%d' % i for i in range(1,6)]
    for col in fields:
        temp.append(pd.pivot_table(error_count,
                                                   index='datetime',
                                                   columns='machineID',
                                             values=col).rolling(window=24).sum().resample('3H', 
                                                        closed='left', label='right').first().unstack())


    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    # error_count.reset_index(inplace=True)#To be activate
    error_count = error_count.dropna()

    # Days Since Last Replacement from Maintenance
    import numpy as np

    # Create a column for each error type
    comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

    # Combine repairs for a given machine in a given hour
    comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

    # Add timepoints where no components were replaced
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                        on=['datetime', 'machineID'],
                                         how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])


    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        # Convert indicator to most recent date of component change
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(),'datetime']

        # Forward-fill the most-recent date of component change
        comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

    # Remove dates in 2014 (may have NaN or future component change dates)    
    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

    # Replace dates of most recent component change with days since most recent component change
    for comp in components:
        comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]) / np.timedelta64(1, 'D')


    #Machine Features
    final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on=['machineID'], how='left')

    #Preparation for prediction
    X = final_feat.drop(['datetime', 'machineID'], 1)
    X_final = pd.get_dummies(X)
    X_final_train = X_final.values

    import pickle 
    # load the model from disk
    model = pickle.load(open(finalized_model, 'rb'))
    prediction = model.predict(X_final_train)
    
    return prediction


# In[8]:


import pickle 
print(pickle.format_version)

