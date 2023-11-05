# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:39:29 2021

@author: mpica
"""
# 0. import packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
#import lightgbm as lgb
    
# models
import sklearn as sk
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LassoCV,Lasso, LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, chi2, f_regression
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

#import xgboost as xgb


%matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
# 1. Read data
df_train = pd.read_csv('C:/Users/mpica/Pictures/Documentos/kaggle/titanic/train.csv')
df_test = pd.read_csv('C:/Users/mpica/Pictures/Documentos/kaggle/titanic/test.csv')

#%%
# 2. Explore data
print(df_train.dtypes)
print(df_train.head())
print(df_train.isna().sum())
print(df_train.describe())
print(df_train.Embarked.value_counts())

target_col = 'Survived'

#%%
# 3. Visulaize data
# sns.distplot(df_train[df_train.Survived==1].Fare, bins = 20, color='green', kde=False)
# sns.distplot(df_train[df_train.Survived==0].Fare, bins = 20, color='orange', kde=False)
sns.displot(x=df_train.Fare, kind='hist', hue = df_train.Sex, bins =20, kde = True, multiple='dodge')

sns.boxplot(x=df_train.Survived, y = df_train.Age)
   
sns.scatterplot(x=df_train.Fare, y = df_train.Age)

       
#%%
# 4. Data cleaning and new features

# impute NA in categorical
col_objects = df_train.columns[df_train.dtypes == 'object']
col_num = df_train.columns[df_train.dtypes != 'object']
for col in col_objects:
    df_train.loc[df_train[col].isna(),col] = 'None'
    df_test.loc[df_test[col].isna(),col] = 'None'

# new numerical from categorical
df_train['new_male'] = df_train['Sex'].map(lambda x: x in ['male']).astype('int64')
df_train['new_floor'] = df_train['Cabin'].str[0] # first letter of cabin, assumed to be floor
df_train['new_hasCabin'] = (df_train['Cabin'] != 'None').astype('int64')
# df_train['new_title'] = df_train.Name.map(lambda x: x.split(',')[1].split(' ')[1])
df_train['new_title'] = df_train.Name.map(lambda x: x.split(',')[1].split('.')[0][1:])
df_train['new_crew'] = df_train.new_title.map(lambda x: x in ['Capt','Col','Major','Rev']).astype('int64')
df_train['new_royalty'] = df_train.new_title.map(lambda x: x in ['Jonkheer','Lady','Sir','the Countess','Don','Dona']).astype('int64') 
df_train['new_noAge'] = (df_train['Age'].isna()).astype('int64')
# df_train['Fare'] = np.log(df_train['Fare']+0.1)
count_floor = df_train.groupby('new_floor').size().reset_index()
count_floor.columns = ['new_floor', 'count_floor']
df_train = pd.merge(df_train, count_floor)

# impute NA in numerical, only Age
auxAge = df_train.groupby('new_title').apply(lambda df: pd.Series({
            'count':len(df['Age']),
            'meanAge':np.mean(df['Age']),
            'medianAge':np.median(df['Age']),
            'isnaAge':df['Age'].isna().sum()
            }))
auxFare = df_train.groupby('Pclass').apply(lambda df: pd.Series({
            'count':len(df['Fare']),
            'meanFare':np.mean(df['Fare']),
            'medianFare':np.median(df['Fare']),
            'isnaFare':df['Fare'].isna().sum()
            }))


df_train.loc[df_train.Age.isna(),'Age'] = df_train.new_title.map(lambda x: auxAge.meanAge.values[auxAge.index==x][0])[df_train.Age.isna()].astype('float64') 

# for col in list(set(col_num)-set([target_col])):
#     print(col)
#     df_train.loc[df_train[col].isna(),col] = df_train[col].median()
#     df_test.loc[df_test[col].isna(),col] = df_train[col].median() # use meadian of train, no leckeage
    
# df_train = df_train.dropna()

OneHotEncode = OneHotEncoder(sparse_output=False)
col_num = list(set(df_train.columns[df_train.dtypes != 'object']) - set(['PassengerId']))
col_num = [i for i in col_num if i != target_col]
ordEncode_col = ['Embarked']
ordEncoded_train = pd.DataFrame(OneHotEncode.fit_transform(df_train[ordEncode_col]))
ordEncoded_train.columns = [ordEncode_col[0]+str('_')+str(i) for i in OneHotEncode.categories_[0]]
#ordEncoded_train.columns = [ordEncode_col[0]+str(i) for i in ordEncoded_train.columns]
ordEncoded_train.index = df_train.index

X_train = pd.concat([df_train[col_num], ordEncoded_train],axis=1)
y_train = df_train[target_col].astype(int)

#%%
# 5. try and validate multiple models and parameters


# model: random forest, test number of trees and minimum number of elements in leafs
scores_all = []
for n_estimators in [10,50,100,200,500]:
    scores = cross_val_score(RandomForestClassifier(n_estimators = n_estimators, min_samples_leaf=10, random_state=0),
                            X_train,y_train,cv=10,scoring='roc_auc') # scoring = 'roc_auc'
    scores_all.append(np.mean(scores))
print(scores_all)
    
scores_all = []
for min_leaf in [1,2,5]:
    scores = cross_val_score(RandomForestClassifier(n_estimators = 100, min_samples_leaf=min_leaf, random_state=0),
                            X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all) 
print(cross_val_score(RandomForestClassifier(n_estimators = 100, min_samples_leaf=2, random_state=0),
                        X_train,y_train,cv=10,scoring='roc_auc').mean())

model = RandomForestClassifier(n_estimators = 100, min_samples_leaf=2, random_state=0)
model.fit(X_train,y_train)
sns.barplot(x = model.feature_importances_, y = model.feature_names_in_)

GridSearchRF=GridSearchCV(estimator=RandomForestClassifier(random_state=0), 
                          param_grid={'n_estimators':[50,100,200],
                                      'min_samples_leaf':[2,5],
                                      'criterion':['gini','entropy']}, 
                          scoring='roc_auc',cv=3)
score=cross_val_score(GridSearchRF,X_train,y_train,scoring='roc_auc',cv=10)

# model: gradient boosting, test number of trees and minimum number of elements in leafs
scores_all = []
for n_estimators in [50,100,150,200]:
    scores = cross_val_score(GradientBoostingClassifier(n_estimators = n_estimators, random_state=0),
                            X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all) 

scores_all = []
for learning_rate in [0.05,0.1,0.3,0.5]:
    scores = cross_val_score(GradientBoostingClassifier(n_estimators = 200, random_state=0, learning_rate = learning_rate),
                            X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all) 

print(cross_val_score(GradientBoostingClassifier(n_estimators = 200, random_state=0, learning_rate = 0.1),
                            X_train,y_train,cv=10,scoring='roc_auc').mean())

# model hist gradient boosting
model = HistGradientBoostingClassifier(loss = 'log_loss', 
                                       max_iter = 1000,
                                       learning_rate = 0.05, 
                                       early_stopping= True,
                                       n_iter_no_change = 100,
                                       validation_fraction = 0.1,
                                       verbose = 1,
                                       random_state = 1)
cross_val_score(model,X_train,y_train,cv=10,scoring='roc_auc').mean()
model.fit(X_train,y_train)

# model support vector machines, test violation penalization and rbf kernel gamma
svm = make_pipeline(StandardScaler(),SVC(random_state=0))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator = svm, 
                   param_grid=PSVM, scoring='roc_auc', cv=2)
score = cross_val_score(GSSVM, X_train.astype(float), y_train,scoring='roc_auc', cv=5).mean()

# model: lasso, test penalization parameter alpha
scores_all = []
for alpha in np.logspace(-5, 1, 7):
    scores = cross_val_score(Lasso(random_state=0,alpha=alpha),X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all) 

scores = cross_val_score(LassoCV(),X_train,y_train,cv=10,scoring='roc_auc')
print(np.mean(scores)) 
model = LassoCV(random_state=0, cv=10)
model.fit(X_train,y_train)

# test several models together
models = {'randomforest': RandomForestClassifier(n_estimators = 100, min_samples_leaf=2, random_state=0),
          'boosting tree': GradientBoostingClassifier(n_estimators = 100, loss = 'exponential', random_state=0),
          'lasso': LassoCV(cv=10,random_state=0),
          'logistic': LogisticRegression(random_state=0)}
scores_all = []
for model_name in models:
    scores = -1*cross_val_score(models[model_name],X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all)

# model: xgboost
xgb.cv(nfold = 10, 
       dtrain = xgb.DMatrix(X_train,label=y_train),
       metrics = ['auc','logloss'],
       params = {'booster': 'gbtree',
                 'learning_rate': 0.3,
                 'max_depth':3,
                 'min_child_weight':10},
       num_boost_round = 100,
       early_stopping_rounds=50).mean()

# n_train = int(X_train.shape[0]*0.95)
# cv_scores = []
# for i in range(10):
#     shuffle = np.random.choice(X_train.shape[0],X_train.shape[0], replace=False)
#     idx_train, idx_test = shuffle[:n_train], shuffle[n_train:]
#     model = xgb.XGBClassifier(objective = 'binary:logistic')
#     model.fit(X_train.loc[idx_train,:],y_train[idx_train])

# lightgbm
def feval_auc(y_pred, lgb_train):
    return 'auc', roc_auc_score(lgb_train.get_label(), y_pred), True
model = lgb.cv(params = {'learning_rate': 0.005,'metric':'auc','objective': 'binary','boosting': 'gbdt','verbosity': 0,'n_jobs': -1,'force_col_wise':True}  ,
               train_set = lgb.Dataset(X_train, y_train),
               nfold = 10,
               num_boost_round = 1000,
               feval = feval_auc,
               callbacks = [lgb.early_stopping(stopping_rounds = 300, verbose = True), lgb.log_evaluation(period = 100)])


#%%
# 6. Submission
df_test['new_male'] = df_test['Sex'].map(lambda x: x in ['male']).astype('int64')
df_test['new_floor'] = df_test['Cabin'].str[0] # first letter of cabin, assumed to be floor
df_test['new_hasCabin'] = (df_test['Cabin'] == 'None').astype('int64')
df_test['new_title'] = df_test.Name.map(lambda x: x.split(',')[1].split('.')[0][1:])
df_test['new_crew'] = df_test.new_title.map(lambda x: x in ['Capt','Col','Major','Rev']).astype('int64')
df_test['new_royalty'] = df_test.new_title.map(lambda x: x in ['Jonkheer','Lady','Sir','the Countess','Don','Dona']).astype('int64') 
df_test['new_noAge'] = (df_test['Age'].isna()).astype('int64')
df_test.loc[df_test.Age.isna(),'Age'] = df_test.new_title[df_test.Age.isna()].map(lambda x: auxAge.meanAge.values[auxAge.index==x][0]).astype('float64')
df_test.loc[df_test.Fare.isna(),'Fare'] = df_test.Pclass[df_test.Fare.isna()].map(lambda x: auxFare.medianFare.values[auxFare.index==x][0]).astype('float64')

ordEncoded_test = pd.DataFrame(OneHotEncode.transform(df_test[ordEncode_col]))
ordEncoded_test.columns = [ordEncode_col[0]+str(i)[2:] for i in OneHotEncode.get_feature_names()]
ordEncoded_test.index = df_test.index

X_test = pd.concat([df_test[col_num], ordEncoded_test],axis=1)

model = GradientBoostingClassifier(n_estimators = 50, random_state=0, learning_rate = 0.1)
model.fit(X_train,y_train)

pd.DataFrame({'PassengerId': df_test['PassengerId'].astype('int'), 'Survived':model.predict(X_test)}).to_csv('C:/Users/mpica/Pictures/Documentos/kaggle/titanic/submission.csv', index = False)

#%%
# 7. test data analysis & check stuff

# 7.1 data aggregation
# check ordinal encoder
aux = pd.concat([df_train, ordEncoded_train],axis=1)
res = aux.groupby(['Embarked']).apply(lambda df: pd.Series({
    'Cmin':min(df['Embarked_C']),
    'Cmax':max(df['Embarked_C']),
    'Nonemin':min(df['Embarked_None']),
    'Nonemax':max(df['Embarked_None']),
    'Qmin':min(df['Embarked_Q']),
    'Qmax':max(df['Embarked_Q']),
    'Smin':min(df['Embarked_S']),
    'Smax':max(df['Embarked_S'])
    }))
aux.groupby(['Embarked']).Embarked_C.agg([min,len,'mean'])
aux.groupby(['Embarked']).agg({'Embarked_C':[min,len,'mean'],'Embarked_None':[min,len,'mean']})

aux.groupby(['Pclass']).apply(lambda df: pd.Series({
    'Fare_mean':np.mean(df['Fare']),
    'Fare_std':np.std(df['Fare'])
    }))

aux.groupby(['Pclass','Embarked']).agg({
    'PassengerId':['count'],
    'Fare':['mean','median','std'],
    'Survived':['mean','std',]})

aux.groupby(['Survived']).Pclass.agg(['value_counts'])
aux.groupby(['Survived','Pclass']).size()

# 7.2 statistical tests
# 7.2.1 two discrete variables: pearson chi2
res = aux.groupby(['Survived']).apply(lambda df: pd.Series({
    'Pclass1':sum(df['Pclass']==1),
    'Pclass2':sum(df['Pclass']==2),
    'Pclass3':sum(df['Pclass']==3),
    'total':df['Pclass'].count()
    }))
# better with crosstab
obs = np.array(pd.crosstab(aux['Survived'],aux['Pclass']))
sts,p,dof,exp = stats.chi2_contingency(pd.crosstab(aux['Survived'],aux['Pclass']))
stats.chisquare(obs.flatten(),exp.flatten(),dof) # test
1-stats.chi2.cdf(sts,dof) # test
print(sts == np.sum((obs.flatten()-exp.flatten())**2/exp.flatten()))

# 7.2.2 continious-discrete variables: Wald same mean, Welch t-test

aux.groupby(['Survived']).Fare.agg(['mean','std','count'])
se = np.sqrt(np.std(aux.Fare[aux.Survived==0])**2/sum(aux.Survived ==0)+
             np.std(aux.Fare[aux.Survived==1])**2/sum(aux.Survived ==1))
sts = np.abs(np.mean(aux.Fare[aux.Survived==0])-np.mean(aux.Fare[aux.Survived==1]))/se
dof = int(se**4/(np.std(aux.Fare[aux.Survived==0])**4/sum(aux.Survived ==0)**3+
             np.std(aux.Fare[aux.Survived==1])**4/sum(aux.Survived ==1)**3))
print(sts,(1-stats.norm.cdf(sts))*2,(1-stats.t.cdf(sts,dof))*2,
      stats.ttest_ind(aux.Fare[aux.Survived==1],aux.Fare[aux.Survived==0], equal_var=False))

se = np.sqrt(np.std(aux.Parch[aux.Survived==0])**2/sum(aux.Survived ==0)+
             np.std(aux.Parch[aux.Survived==1])**2/sum(aux.Survived ==1))
sts = np.abs(np.mean(aux.Parch[aux.Survived==0])-np.mean(aux.Parch[aux.Survived==1]))/se
dof = int(se**4/(np.std(aux.Parch[aux.Survived==0])**4/sum(aux.Survived ==0)**3+
             np.std(aux.Parch[aux.Survived==1])**4/sum(aux.Survived ==1)**3))
print(sts,(1-stats.norm.cdf(sts))*2,(1-stats.t.cdf(sts,dof))*2,
      stats.ttest_ind(aux.Parch[aux.Survived==1],aux.Parch[aux.Survived==0], equal_var=False))

se = np.sqrt(np.std(aux.Age[aux.Survived==0])**2/sum(aux.Survived ==0)+
             np.std(aux.Age[aux.Survived==1])**2/sum(aux.Survived ==1))
sts = np.abs(np.mean(aux.Age[aux.Survived==0])-np.mean(aux.Age[aux.Survived==1]))/se
dof = int(se**4/(np.std(aux.Age[aux.Survived==0])**4/sum(aux.Survived ==0)**3+
             np.std(aux.Age[aux.Survived==1])**4/sum(aux.Survived ==1)**3))
se0 = np.std(aux.Age[aux.Survived==0])
se1 = np.std(aux.Age[aux.Survived==1])
n0 = sum(aux.Survived ==0)
n1 = sum(aux.Survived ==1)
dof = (se0**2/n0 + se1**2/n1)**2/(se0**4/n0**2/(n0-1) + se1**4/n1**2/(n1-1))
print(sts,(1-stats.norm.cdf(sts))*2,(1-stats.t.cdf(sts,dof))*2,
      stats.ttest_ind(aux.Age[aux.Survived==1],aux.Age[aux.Survived==0], equal_var=False))

# better with the p-value of coefficient in univariate regression statistics (see also next)
f,p = f_classif(X_train,y_train)
chi,p = chi2(X_train,y_train)

# also cutting variables
pd.crosstab(aux['Survived'],pd.cut(aux['Fare'],[-1000,0,10,25,50,100,1000]))

# 7.2.3: continuous-continuous variables: pearson correlation test

n = 100
corr_bootstrap = [np.corrcoef(aux.loc[np.random.choice(len(aux),len(aux),replace=True),['Fare','Age']].T)[0,1] for i in range(n)]
r = stats.pearsonr(aux.Fare,aux.Age)[0] # correlation
print(np.mean(corr_bootstrap), np.std(corr_bootstrap),
      stats.norm.cdf(-np.mean(corr_bootstrap)/np.std(corr_bootstrap)*np.sqrt(n))*2,
      (r,2*(1-stats.t.cdf(r*np.sqrt(len(aux)-2)/np.sqrt(1-r**2),len(aux)-2))),
      stats.pearsonr(aux.Fare,aux.Age))

print(stats.ttest_ind(aux.Fare,aux.Age))

# using models
model = smf.glm(formula = 'Fare ~ Age ', data = aux, family=sm.families.Gaussian()).fit()

# 7.3 statistical inference on linear models
X_train_pIndep = sm.add_constant(X_train)
model = sm.GLM(y_train, X_train_pIndep, family=sm.families.Binomial()).fit()
model.summary()
model2 = sm.Logit(y_train, X_train_pIndep).fit()

Xy_train = X_train.copy()
Xy_train['Survived'] = y_train
model1p = smf.glm(formula = 'Survived ~  Pclass + Age + SibSp + Parch + Fare + C(new_male) + Embarked_C+Embarked_None+Embarked_Q', 
                 data = Xy_train, family=sm.families.Binomial()).fit()
model2p = smf.logit(formula = 'Survived ~  Pclass + Age + SibSp + Parch + Fare + C(new_male) + Embarked_C+Embarked_None+Embarked_Q', 
                 data = Xy_train).fit()
model1pp = smf.glm(formula = 'Survived ~  Pclass + Age + SibSp + Parch + Fare + C(new_male) + C(Embarked)', 
                 data = df_train, family=sm.families.Binomial()).fit()
#model2p = smf.logit(formula = 'Survived ~  Embarked_C+Embarked_None+Embarked_Q', 
                 #data = Xy_train).fit()
print(2*(model2p.llf-model2p.llnull), model2p.llr, model2p.llr_pvalue,\
      1-stats.chi2.cdf(model2p.llr,model2p.df_model)) #

# y_pred = model.predict(X_train_pIndep)
# model.resid_response = (y_pred- y_train)
# model.resid_pearson = (y_pred- y_train)/np.sqrt(y_pred*(1-y_pred))
# model.pearson_chi2 = sum(model.resid_pearson**2)
# model.deviance = -1*sum(np.log(y_pred[y_train==1]))*2+sum(np.log(1-y_pred[y_train==0]))*2 # = sum(model.resid_deviance**2)
# model.llf = -1*model.deviance/2
# pseudo-R**2 = 1-LL/LL_0

# 7.4 random generation
# random integer, for bootstrapping
stats.randint.rvs(0,1000,size=10)
np.random.choice(1000,size=10) #replace = True
np.random.choice(1000,size=10,replace=False)
# random uniform -1 1
stats.uniform.rvs(-1,1,size=10)
np.random.uniform(-1,1,10)
# random normal

# 7.5 cluster features
inertias = []
for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train[['Pclass','Fare','Age','Embarked_C']])
    inertias.append(kmeans.inertia_)
plt.plot(range(2,10),inertias)

from itertools import permutations, combinations