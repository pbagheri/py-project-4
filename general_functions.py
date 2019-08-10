# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:19:04 2017

@author: Payam
"""

import datetime

# function for reading csv files
def readfile(filnam):
    path = 'C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/csv/' + filnam + '.csv'
    print(path)
    exec(filnam + '= pd.read_csv(path);' + filnam + '.head()')
    #return dfname
    

# function for replacing '.' with np.nan
def nanputter(dfname):
    cuscol = dfname.columns
    for i in cuscol:
        dfname[i].replace('.',np.nan, inplace = True)
        

# function for replacing character levels in categorical variables with numbers
def chartonum(dfname, colname):
    keys = {}
    for i in range(len(dfname[colname].unique())):
        keys[dfname[colname].unique()[i]] = i+1
    print(keys)
    dfname[colname].replace(keys, inplace = True)
    return dfname[colname]

# function for replacing missing value with a user-indicated value
def repl_mis(dfname,colname,val):
    dfname[colname].replace({np.nan:val}, inplace = True)
    return dfname[colname]

# function to calculate age from birth-date
def datetoage(date): 
    months = { 'JAN' : 1, 'FEB' : 2, 'MAR':3, 'APR': 4, 'MAY': 5, 'JUN': 6, \
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT':10, 'NOV': 11, 'DEC': 12}
    ldate = list(date); ldate
    day = int(''.join(ldate[0:2])); day
    mon = months[''.join(ldate[2:5])]; mon
    year = int(''.join(ldate[5:])); year
    curdate = datetime.datetime(day=day, month=mon, year=year).date()
    lastdate = datetime.datetime.strptime('2006-01-01', "%Y-%m-%d").date()
    delta =  (lastdate - curdate).days
    age = delta/365.0
    return age

"""
# Function for turning date in a certain format to a numerical value
def datenum(date): 
    months = { 'JAN' : 1, 'FEB' : 2, 'MAR':3, 'APR': 4, 'MAY': 5, 'JUN': 6, \
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT':10, 'NOV': 11, 'DEC': 12}
    ldate = list(date); ldate
    day = int(''.join(ldate[0:2])); day
    mon = months[''.join(ldate[2:5])]; mon
    year = int(''.join(ldate[5:])); year
    curdate = datetime.datetime(day=day, month=mon, year=year).date()
    return curdate
"""

# Function for doing gridsearch for finding optimal parameters for the model
def gridsearch_model(feat, targ):
    train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3, random_state=10)
        
    tuned_parameters = [{'class_weight' : [{0:1, 1:1}, {0:0.5, 1:1}, {0:1, 1:0.5},\
                       {0:0.5, 1:2}, {0:2, 1:0.5}, {0:0.5, 1:3}, {0:3, 1:0.5}]}]
    
    scores = ['roc_auc']
        
    for score in scores:
        print("Tuning hyper-parameters for %s" % score)
    
        model = GridSearchCV(LogisticRegression(), tuned_parameters, cv=None, scoring= score)
        model.fit(train, train_target)
    
        print("Best parameters set found on development set:")
        print(model.best_params_)
        print("Grid scores on development set:")
        for params, mean_score, scores in model.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
            
        print("Detailed classification report:")
        print("The model is train_baled on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = test_target, model.predict(test)
        print(classification_report(y_true, y_pred))
        sc = model.predict_proba(test)
        print('auc is', roc_auc_score(test_target, sc[:,1]))
            
# Function for doing gridsearch for finding optimal parameters for the lasso
def gridsearch_lasso(feat, targ):
    train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3, random_state=10)

    tuned_parameters = [{'C': [2, 1, 5e-1, 4e-1, 3e-1, 3e-1, 1e-1, 0.5e-1],\
    'class_weight' : [{0:1, 1:1}, {0:0.5, 1:1}, {0:1, 1:0.5}, {0:0.5, 1:2}, \
    {0:2, 1:0.5}]}]
            
    scores = ['roc_auc']
        
    for score in scores:
        print("Tuning hyper-parameters for %s" % score)
    
        model = GridSearchCV(LogisticRegression(penalty='l1'), tuned_parameters, cv=None, scoring= score)
        model.fit(train, train_target)
    
        print("Best parameters set found on development set:")
        print(model.best_params_)
        print("Grid scores on development set:")
        for params, mean_score, scores in model.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
            
        print("Detailed classification report:")
        print("The model is train_baled on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = test_target, model.predict(test)
        print(classification_report(y_true, y_pred))
        sc = model.predict_proba(test)
        print('auc is', roc_auc_score(test_target, sc[:,1]))


# Function for checking multicoliearity. The output is a list of feature names
# that have no multicolinearity between themselves
def multicolcheck(arr,rem_col):
    try:
        corr=np.corrcoef(arr,rowvar=0)
        eigval,eigvec=np.linalg.eig(corr)
        lowval = [i for i in range(len(eigval)) if eigval[i] < 0.1]
        eigval_list = list(eigval)
        eigenvec_low = eigvec[:, lowval]
            
        lowcols = []
        for i in range(eigenvec_low.shape[0]):
            if abs(eigenvec_low[i]).max() < 0.1:
                lowcols.append(i)
        
        collin = list(set(range(len(rem_col))) - set(lowcols)) 
        lowcols.append(collin[0])
        nocollin = lowcols
        noncollin = [rem_col[i] for i in nocollin]
        return noncollin
    except ValueError:
        print('No multicolinearity')
        

# Function for applying the lasso feature selection
def lassoselection(feat,targ, C, class_weight):
    train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3)

    lgr = LogisticRegression(penalty='l1', C= C, class_weight = class_weight)
    lgr.fit(train,train_target)
    pred = lgr.predict(test)
    print(classification_report(test_target, pred))
    print('accuracy_score is', accuracy_score(test_target, pred, normalize=True, sample_weight=None))
    scores = lgr.predict_proba(test)
    print('roc_auc_score is', roc_auc_score(test_target, scores[:,1]))
    lgrcoef = lgr.coef_
    return lgr
    

# Function for applying the model
def modelapplication(feat,targ, class_weight):
    train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3)

    lgr = LogisticRegression(class_weight = class_weight)
    lgr.fit(train,train_target)
    pred = lgr.predict(test)
    print(classification_report(test_target, pred))
    print('accuracy_score is', accuracy_score(test_target, pred, normalize=True, sample_weight=None))
    scores = lgr.predict_proba(test)
    print('roc_auc_score is', roc_auc_score(test_target, scores[:,1]))
    lgrcoef = lgr.coef_
    return lgr

# Function for obtaining the AUC score of model application
def modelapplication_auc(feat,targ, class_weight):
    train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3)

    lgr = LogisticRegression(class_weight = class_weight)
    lgr.fit(train,train_target)
    pred = lgr.predict(test)
    print(classification_report(test_target, pred))
    print('accuracy_score is', accuracy_score(test_target, pred, normalize=True, sample_weight=None))
    scores = lgr.predict_proba(test)
    print('roc_auc_score is', roc_auc_score(test_target, scores[:,1]))
    lgrcoef = lgr.coef_
    return roc_auc_score(test_target, scores[:,1])


# Function for calculating the gain, lift, decile and KS chart
def calc_lift(x,y,clf,bins=10):
    #Actual Value of y
    y_actual = y
    #Predicted Probability that y = 1
    y_prob = clf.predict_proba(x)
    #Predicted Value of Y
    y_pred = clf.predict(x)
    cols = ['ACTUAL','PROB_EVENT','PREDICTED']
    data = [y_actual,y_prob[:,1],y_pred]
    df = pd.DataFrame(dict(zip(cols,data)))
    
    #Observations where y=1
    total_events_n = df['ACTUAL'].sum()    
    #Total Observations
    total_n = df['ACTUAL'].count()
    total_non_events_n = total_n - total_events_n
    random_event_prob = total_events_n/float(total_n)

    #Create Bins where First Bin has Observations with the
    #Highest Predicted Probability that y = 1
    df['BIN_EVENT'] = pd.qcut(df['PROB_EVENT'],bins,labels=False)
    
    pos_group_df = df.groupby('BIN_EVENT')
    #Percentage of Observations in each Bin where y = 1
    num_total = pos_group_df['ACTUAL'].count()
    num_events = pos_group_df['ACTUAL'].sum()
    num_non_events = num_total - num_events
    percent_events = 100*num_events/total_events_n
    percent_non_events = 100*num_non_events/total_non_events_n
    #lift_index_positive = (percent_events/random_event_prob)*100
    
    #Consolidate Results into Output Dataframe
    liftdf = pd.DataFrame({'NUM_TOTAL':num_total, 'NUM_EVENTS':num_events, 
                           'NUM_NON_EVENTS':num_non_events, 'PERCENT_EVENTS':percent_events,
                            'PERCENT_NON_EVENTS':percent_non_events,'RANDOM_NUM_EVENT': total_events_n/10})
    
    liftdf.loc[10] = 0     
    liftdf['BIN_EVENT'] = liftdf.index
    liftdf = liftdf.sort_values('BIN_EVENT', ascending = False)
    liftdf.drop('BIN_EVENT', axis=1)
    liftdf['PERCENTS'] = range(0,110,10)
    liftdf['DECILE_BIN'] = liftdf['PERCENTS']/10
    liftdf['CUM_GAIN'] = [liftdf['PERCENT_EVENTS'][0:i+1].sum() for i in range(11)]
    liftdf['CUM_NEG'] = [liftdf['PERCENT_NON_EVENTS'][0:i+1].sum() for i in range(11)]
    liftdf['CUM_LIFT'] = liftdf['CUM_GAIN']/liftdf['PERCENTS']
    liftdf['DECILE_LIFT'] = liftdf['NUM_EVENTS']/liftdf['RANDOM_NUM_EVENT']
    liftdf['DECILE_BASE'] = liftdf['PERCENTS']/liftdf['PERCENTS']
    return liftdf


def feat_decile(x,f,clf,bins=10):
    #Actual Value of y
    feat = f
    y_prob = clf.predict_proba(x)
    cols = ['feat','PROB_EVENT']
    data = [feat,y_prob[:,1]]
    df = pd.DataFrame(dict(zip(cols,data)))
    
    df['BIN_EVENT'] = pd.qcut(df['PROB_EVENT'],bins,labels=False)
    pos_group_df = df.groupby('BIN_EVENT')
    feat_mean = pos_group_df['feat'].mean()
    
    featdf = pd.DataFrame({'feat_mean':feat_mean})
    
    featdf['BIN_EVENT'] = featdf.index
    featdf = featdf.sort_values('BIN_EVENT', ascending = False)
    featdf.drop('BIN_EVENT', axis=1)
    featdf['DECILE_BIN'] = range(1,11)
    return featdf


