import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

# DATA ACQUISITION
def load_db_to_df(db_name):
    table_name_1  = 'data'
    q1 = 'SELECT * FROM {tn}'.format(tn=table_name_1)
    conn = sqlite3.connect(db_name)
    data = pd.read_sql_query(q1, conn, index_col=['id'])
    conn.close()
    return data

def show_db_tables(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    query_check  = 'SELECT name FROM sqlite_master WHERE type= "table";'
    c.execute(query_check)
    table_set = [item[0] for item in c.fetchall()]
    print(table_set)


def save_tables(db_name, t1, t2, t3):
    conn = sqlite3.connect(db_name)
    t1.to_sql("data", conn, if_exists="replace")
    t2.to_sql("dayta", conn, if_exists="replace")
    t3.to_sql("week_data", conn, if_exists="replace")
    conn.commit()
    conn.close()
    print('Tables Saved to',db_name)




# EDA

# gather_weather_data_per_hour takes the data, groups it by the hour and displays two kde plots in one figure, one grouped by the mean, the other one grouped by the median.
def gather_weather_data_per_hour(data):
    fig = plt.figure(figsize=(20,10))

    axes1 = fig.add_subplot(2,2,1)
    temp_hour = data.groupby('hour')['temp'].median()
    temp_hourm = data.groupby('hour')['temp'].mean()
    sns.kdeplot(temp_hour, shade=True, color="coral", label='grouped by median')
    plt.vlines(temp_hour.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='orange', label='mean')
    plt.vlines(temp_hour.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="red", label='median')
    sns.kdeplot(temp_hourm, shade=True, color="royalblue", label='grouped by mean')
    plt.vlines(temp_hourm.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='blue', label='mean')
    plt.vlines(temp_hourm.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="m", label='median')
    plt.title('temperature')
    plt.legend()
    axes2 = fig.add_subplot(2,2,2)
    atemp_hour = data.groupby('hour')['atemp'].median()
    atemp_hourm = data.groupby('hour')['atemp'].mean()
    sns.kdeplot(atemp_hour, shade=True, color="coral", label='grouped by median')
    plt.vlines(atemp_hour.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='orange')
    plt.vlines(atemp_hour.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="red")
    sns.kdeplot(atemp_hourm, shade=True, color="royalblue", label='grouped by mean')
    plt.vlines(atemp_hourm.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='blue')
    plt.vlines(atemp_hourm.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="m")
    plt.title('temperature felt')
    axes3 = fig.add_subplot(2,2,3)
    wind_hour = data.groupby('hour')['windspeed'].median()
    wind_hourm = data.groupby('hour')['windspeed'].mean()
    sns.kdeplot(wind_hour, shade=True, color="coral", label='grouped by median')
    plt.vlines(wind_hour.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='orange')
    plt.vlines(wind_hour.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="red")
    sns.kdeplot(wind_hourm, shade=True, color="royalblue", label='grouped by mean')
    plt.vlines(wind_hourm.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='blue')
    plt.vlines(wind_hourm.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="m")
    plt.title('windspeed')
    axes4 = fig.add_subplot(2,2,4)
    hum_hour = data.groupby('hour')['humidity'].median()
    hum_hourm = data.groupby('hour')['humidity'].mean()
    sns.kdeplot(hum_hour, shade=True, color="coral", label='grouped by median')
    plt.vlines(hum_hour.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='orange')
    plt.vlines(hum_hour.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="red")
    sns.kdeplot(hum_hourm, shade=True, color="royalblue", label='grouped by mean')
    plt.vlines(hum_hourm.mean(),     # Plot black line at mean
               ymin=0,
               ymax=6,
               linewidth=5.0, color='blue')
    plt.vlines(hum_hour.median(),   # Plot red line at median
               ymin=0,
               ymax=6,
               linewidth=2.0,
               color="m")
    plt.title('humidity')
    plt.show()
#    print('SkewnessTemp\nGroupByMean:',temp_hourm.skew(),'\nGroupbyMedian:', temp_hour.skew())
#    print('Kurtosis\nGroupByMean:',temp_hourm.kurt(),'\nGroupbyMedian:', temp_hour.kurt())
#    print('SkewnessATemp\nGroupByMean:',atemp_hourm.skew(),'\nGroupbyMedian:', atemp_hour.skew())
#    print('Kurtosis\nGroupByMean:',atemp_hourm.kurt(),'\nGroupbyMedian:', atemp_hour.kurt())
#    print('SkewnessWind\nGroupByMean:',wind_hourm.skew(),'\nGroupbyMedian:', wind_hour.skew())
#    print('Kurtosis\nGroupByMean:',wind_hourm.kurt(),'\nGroupbyMedian:', wind_hour.kurt())
#    print('SkewnessHum\nGroupByMean:',hum_hourm.skew(),'\nGroupbyMedian:', hum_hour.skew())
#    print('Kurtosis\nGroupByMean:',hum_hourm.kurt(),'\nGroupbyMedian:', hum_hour.kurt())

# Function plotting the kde plots with their mean and median.
def get_distribution_with_center_metrics(weeta):
    plt.figure(figsize=(15,4))
    sns.kdeplot(weeta.cnt, shade=True, color="coral")
    plt.vlines(weeta.mean(),     # Plot black line at mean
               ymin=0,
               ymax=0.004,
               linewidth=5.0, color='orange', label='mean')
    plt.vlines(weeta.median(),   # Plot red line at median
               ymin=0,
               ymax=0.004,
               linewidth=2.0,
               color="red", label='median')
    plt.legend()
    plt.show()


# Fill missing values through interpolation
def fill_missin(dat, kind):
    d = dat.set_index(['date','hour'])
    d.sort_index(inplace=True)
    m = d.unstack().T

    # Fill missing values
    m.interpolate(limit_direction='both', inplace=True)
    if kind == 1:
        m = m.astype(int, inplace=True)
    # Revert to original dataframe
    m = m.T.stack()
    m.reset_index(inplace=True)
    return m


# Normalize hourly records by day
def normalize_data(s, data):
    column_names = 'n'+s.columns[-1][0].upper()+s.columns[-1][1:]
    ## Get the total of bikes per day
    s.set_index(['date','hour'], inplace=True)
    s.sort_index(inplace=True)
    st = s.groupby('date').sum() # Group by day

    s['x'] = s.apply(lambda x:x/st.iloc[:,-1])
    s.reset_index(inplace=True)
    s.drop(s.columns[:3], axis=1, inplace=True)
    s['newIndex'] = s.index.values+1
    s.set_index('newIndex', inplace=True)
    # Rename column
    s.columns = [column_names]
    # Merge normalized column to the dataframe
    return pd.merge(data, s, right_index=True, left_index=True)

# OLS y and the feature set, with const being a boolean indicating
# an OLS with or without intercept.
def processSubset1(feature_set, const, X, y):
    #Our model needs an intercept so we add a column of 1s:
    if const:
        Xo = sm.add_constant(X[list(feature_set)])
    else:
        Xo = X[list(feature_set)]
    model = sm.OLS(y, Xo)
    regr = model.fit()
    RSS = ((regr.predict(Xo) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def processSubset2(feature_set, X_train, y_train, X_test, y_test, const):
    if const:
        Xo = sm.add_constant(X_train[list(feature_set)])
        Xt = sm.add_constant(X_test[list(feature_set)])
    else:
        Xo = X_train[list(feature_set)]
    model = sm.OLS(y_train,X_train[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X_test[list(feature_set)]) - y_test) ** 2).sum()
    return {"model":regr, "RSS":RSS}



# Iterate through the models generated by the processSubset and
# return the best model having the lowest RSS
def getBest(k,c,X,y):
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset1(combo,c,X,y))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    #print('Models:',models)
    # Choose the model with the lowest RSS
    best_model = models.loc[models['RSS'].argmin()]
#    print("Processed ", models.shape[0], "models on", k,"predictor(s)")
    # Return the best model, along with some other useful information about the model
    return best_model



# Forward subset selection method.
def forward(predictors, X_train, y_train, X_test, y_test, const):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]
    results = []
    for p in remaining_predictors:
        results.append(processSubset2(predictors + [p], X_train, y_train, X_test, y_test, const))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the lowest RSS
    best_model = models.loc[models['RSS'].argmin()]
    # Return the best model, along with some other useful information about the model
    return best_model

def print_optimal_number_predictors(models):
    models.index = models.index+1
    models.drop('index', axis=1, inplace=True)

    fig = plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

    # Set up a 2x2 grid so we can look at 4 plots at once
    plt.subplot(2, 2, 1)

    # We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
    # The argmax() function can be used to identify the location of the maximum point of a vector
    plt.plot(models["RSS"])
    plt.xlabel('# Predictors')
    plt.ylabel('RSS')
    plt.plot(models["RSS"].argmin(), models["RSS"].min(), "or")

    # We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
    # The argmax() function can be used to identify the location of the maximum point of a vector

    rsquared_adj = models.apply(lambda row: row[1].rsquared_adj, axis=1)

    plt.subplot(2, 2, 2)
    plt.plot(rsquared_adj)
    plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
    plt.xlabel('# Predictors')
    plt.ylabel('adjusted rsquared')

    # We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
    aic = models.apply(lambda row: row[1].aic, axis=1)

    plt.subplot(2, 2, 3)
    plt.plot(aic)
    plt.plot(aic.argmin(), aic.min(), "or")
    plt.xlabel('# Predictors')
    plt.ylabel('AIC')

    bic = models.apply(lambda row: row[1].bic, axis=1)

    plt.subplot(2, 2, 4)
    plt.plot(bic)
    plt.plot(bic.argmin(), bic.min(), "or")
    plt.xlabel('# Predictors')
    plt.ylabel('BIC')
    return fig

def predict(res, new):
    # Get the predicted values
    fit = pd.DataFrame(res.predict(new), columns=['fit'])
    # Get the confidence interval for the model (and rename the columns to something a bit more useful)
    ci = res.conf_int().rename(columns={0: 'lower', 1: 'upper'})

    # Now a little bit of matrix multiplication to get the confidence intervals for the predictions
    ci = ci.T.dot(new.T).T

    # And finally wrap up the confidence intervals with the predicted values
    return pd.concat([fit, ci], axis=1)


def get_models(mymodels, X, y):
    # Subset selection
    mymodels = get_model_subset_selection(mymodels, X, y)
    # Forward subset selection with training set
    mymodels = get_model_forward_subset_selection(mymodels, X, y)
    # Forward method Using the full dataset
    mymodels = get_model_forward_subset_selection_with_full_dataset(mymodels,X,y)
    # Model selection using the Cross Validation
    mymodels = get_model_cross_validation(mymodels, X, y)

    # Set RSS as float
    mymodels.RSS = mymodels.RSS.astype('float')
    return mymodels

def get_model_subset_selection(mymodels, X, y):
    models = pd.DataFrame(columns=["RSS", "model"])
    for i in range(1, len(X.columns)+1):
        models.loc[i, ["RSS", "model"]] = getBest(i,True,X,y)
    # Merging results
    idx = mymodels.iloc[-1].name
    models.reset_index(inplace=True)
    models.index = models.iloc[:,0].apply(lambda x: x+idx)
    models.drop('index', axis=1, inplace=True)
    mymodels = pd.concat([mymodels, models])
    # Assign Method
    mymodels.loc[idx+1:,'Method'] = 'Slow_Subset_selection'
    return mymodels

def get_model_forward_subset_selection(mymodels, X, y):
    # Forward Subset Selection
    # Training Set and Test Set
    np.random.seed(seed=12)
    train = np.random.choice([True, False], size = len(y), replace = True)
    test = np.invert(train)
    models_train = pd.DataFrame(columns=["RSS", "model"])
    predictors = []
    for i in range(1,len(X.columns)+1):
    #    models_train.loc[i] = forward(predictors, X[train], y[train]["Salary"], X[test], y[test]["Salary"])
        # Pull out predictors we still need to process
        remaining_predictors = [p for p in X[train].columns if p not in predictors]
        results = []
        for p in remaining_predictors:
            results.append(processSubset2(predictors + [p], X[train], y[train], X[test], y[test], True))
        # Wrap everything up in a nice dataframe
        models = pd.DataFrame(results)
        # Choose the model with the lowest RSS
        best_model = models.loc[models['RSS'].argmin()]
        # Return the best model, along with some other useful information about the model
        models_train.loc[i] = best_model
        predictors = models_train.loc[i]["model"].model.exog_names
    # Merging results
    idx = mymodels.iloc[-1].name
    models_train.reset_index(inplace=True)
    models_train.index = models_train.iloc[:,0].apply(lambda x: x+idx)
    models_train.drop('index', axis=1, inplace=True)
    mymodels = pd.concat([mymodels, models_train])
    # Assign Method
    mymodels.loc[idx+1:,'Method'] = 'Fwd_with_train_and_test'
    return mymodels

# Forward method Using the full dataset
def get_model_forward_subset_selection_with_full_dataset(mymodels,X,y):
    models_full = pd.DataFrame(columns=["RSS", "model"])
    predictors = []
    for i in range(1,len(X.columns)+1):
        models_full.loc[i] = forward(predictors, X, y, X, y, True)
        predictors = models_full.loc[i]["model"].model.exog_names
    # Merging results
    idx = mymodels.iloc[-1].name
    models_full.reset_index(inplace=True)
    models_full.index = models_full.iloc[:,0].apply(lambda x: x+idx)
    models_full.drop('index', axis=1, inplace=True)
    mymodels = pd.concat([mymodels, models_full])
    # Assign Method
    mymodels.loc[idx+1:,'Method'] = 'Fwd_with_full_db'
    return mymodels

def get_model_cross_validation(mymodels, X, y):
    k = 10 # number of folds
    np.random.seed(seed=1)
    folds = np.random.choice(k, size = len(y), replace = True)
    # Create a DataFrame to store the results of our upcoming calculations
    cv_errors = pd.DataFrame(columns=range(1,k+1), index=range(1,5))
    cv_errors = cv_errors.fillna(0)
    models_cv = pd.DataFrame(columns=["RSS", "model"])
    # Outer loop iterates over all folds
    for j in range(1,k+1):
        # Reset predictors
        predictors = []
        # Inner loop iterates over each size i
        for i in range(1,len(X.columns)+1):
            # Perform forward selection on the full dataset minus the jth fold, test on jth fold
            models_cv.loc[i] = forward(predictors, X[folds != (j-1)], y[folds != (j-1)], X[folds == (j-1)], y[folds == (j-1)],True)
            # Save the cross-validated error for this fold
            cv_errors[j][i] = models_cv.loc[i]["RSS"]
            # Extract the predictors
            predictors = models_cv.loc[i]["model"].model.exog_names
    # Merging results
    idx = mymodels.iloc[-1].name
    models_cv.reset_index(inplace=True)
    models_cv.index = models_cv.iloc[:,0].apply(lambda x: x+idx)
    models_cv.drop('index', axis=1, inplace=True)
    mymodels = pd.concat([mymodels, models_cv])
    # Assign Method
    mymodels.loc[idx+1:,'Method'] = 'Cross_Validation'
    return mymodels

def get_single_models_temp_atemp(dayta):
    # response=atemp
    mymodels = pd.DataFrame(columns=["Method","RSS", "model"])

    # Single and multiple regressions

    single_reg = smf.ols('atemp ~ temp', data=dayta).fit()
    RSS1 = ((single_reg.predict(dayta.temp) - dayta.atemp) ** 2).sum()

    mlr_all_var = sm.OLS.from_formula('atemp ~ temp + windspeed + humidity', dayta).fit()
    RSS2 = ((mlr_all_var.predict(dayta.loc[:,['temp','windspeed','humidity']]) - dayta.atemp) ** 2).sum()

    mlr_tmp_times_hum = sm.OLS.from_formula('atemp ~ temp * humidity', dayta).fit()
    dayta['tempVhum'] = dayta.loc[:,'temp']*dayta.loc[:,'humidity']
    #RSS3 = ((mlr_tmp_times_hum.predict(dayta.tempVhum) - dayta.atemp) ** 2).sum()


    mlr_atmp_sq_atemp = sm.OLS.from_formula('atemp ~ temp + np.square(temp)', dayta).fit()
    #RSS4 = ((mlr_atmp_sq_atemp.predict(dayta.temp) - dayta.atemp) ** 2).sum()

    mymodels.loc[0] = ['single_reg_with_temp', RSS1, single_reg]
    mymodels.loc[1] = ['mlr_all_vars', RSS2, mlr_all_var]
    mymodels.loc[2] = ['mlr_temp_times_hum', 0.42, mlr_tmp_times_hum]
    mymodels.loc[3] = ['mlr_tmp_sq_temp', 0.42, mlr_atmp_sq_atemp]
    return mymodels

def get_single_models_nCnt_atemp(dayta):
    # response=nCnt
    mymodels = pd.DataFrame(columns=["Method","RSS", "model"])

    # Single and multiple regressions
    single_reg = smf.ols('nCnt ~ atemp', data=dayta).fit()
    RSS1 = ((single_reg.predict(dayta.atemp) - dayta.nCnt) ** 2).sum()

    mlr_all_var = sm.OLS.from_formula('nCnt ~ atemp + temp + windspeed + humidity', dayta).fit()
    RSS2 = ((mlr_all_var.predict(dayta.loc[:,['atemp','temp','windspeed','humidity']]) - dayta.nCnt) ** 2).sum()

    mlr_tmp_times_hum = sm.OLS.from_formula('nCnt ~ temp * humidity', dayta).fit()
    dayta['tempVhum'] = dayta.loc[:,'temp']*dayta.loc[:,'humidity']
    #RSS3 = ((mlr_tmp_times_hum.predict(dayta.tempVhum) - dayta.atemp) ** 2).sum()


    mlr_atmp_sq_atemp = sm.OLS.from_formula('atemp ~ temp + np.square(temp)', dayta).fit()
    #RSS4 = ((mlr_atmp_sq_atemp.predict(dayta.temp) - dayta.atemp) ** 2).sum()

    mymodels.loc[0] = ['single_reg_with_temp', RSS1, single_reg]
    mymodels.loc[1] = ['mlr_all_vars', RSS2, mlr_all_var]
    mymodels.loc[2] = ['mlr_temp_times_hum', np.nan, mlr_tmp_times_hum]
    mymodels.loc[3] = ['mlr_tmp_sq_temp', np.nan, mlr_atmp_sq_atemp]
    return mymodels


def RMSE(predicted, actual):
    mse = (predicted - actual)**2
    rmse = np.sqrt(mse.sum()/mse.count())
    return rmse

# AUGMENTED DICKEY FULLER TEST FOR STATIONARITY
def adf(ts):

    # Determing rolling statistics
    rolmean = ts.rolling(window=2).mean()
    rolstd = ts.rolling(window=2).std()

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original', Linewidth=4, alpha=0.5)
    mean = plt.plot(rolmean, color='black', label='Rolling Mean', linestyle='--')
    std = plt.plot(rolstd, color='orange', label = 'Rolling Std', linestyle='--')
    plt.legend(loc='best')
    plt.grid('on')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Calculate ADF factors
    adftest = adfuller(ts, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput
