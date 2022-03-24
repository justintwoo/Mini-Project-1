# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('CFPB-financial-wellness-data.csv', index_col = 'PUF_ID')

data.head()

# Ignore survey weights; we also drop PRODUSE_6 as well because it is always 1 when PRODUSE_3 is 0
X = data.drop(columns = ["PRODUSE_3", "PRODUSE_6", "finalwt", "IMPUTATION_FLAG", "sample"]) 
y = data.PRODUSE_3

selector = SelectKBest(f_classif, k = 20)

X_0 = selector.fit_transform(X, y)

X_f = pd.DataFrame(X_0, columns = X.columns[selector.get_support()])
#X_test_f = pd.DataFrame(test_data, columns = X_train.columns[selector.get_support()])

X.columns[selector.get_support()]

# Encode categorical variables 
X_cat = X_f.drop(columns = ["FWBscore", "FSscore", "LMscore", "KHscore", "CONNECT", "LIFEEXPECT"], errors = "ignore").astype('str')
X_numeric = X_f.loc[:, ~X_f.columns.isin(X_cat.columns)]

X_final = pd.concat([pd.get_dummies(X_cat), X_numeric], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y.reset_index(drop = True), test_size = 0.2, random_state = 0)

# scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
# X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

# Drop any constant features in X_train
X_train = X_train.loc[:, X_train.nunique() != 1]
X_test = X_test.loc[:, X_test.columns.isin(X_train.columns)]

pd.Series(X_train.corrwith(y_train)).sort_values()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 100)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = 100, learning_rate = 1)
model.fit(X_train, y_train)


# Best Params - AdaBoostClassifier 
ada= AdaBoostClassifier()
search_grid = {'n_estimators': [100, 150, 200]}
search = GridSearchCV(estimator = ada, param_grid = search_grid, scoring = 'f1')

search.fit(X_train, y_train)
print(search.best_params_) 
print(search.best_score_)

y_test_ppred = search.best_estimator_.predict_proba(X_test)[:,1]
y_test_pred = [(x > 0.5) for x in y_test_ppred]

print(classification_report(y_test, y_test_pred))

from sklearn.metrics import roc_auc_score, roc_curve

fp, tp, t = roc_curve(y_test, y_test_ppred)
auc = roc_auc_score(y_test, y_test_ppred)
plt.plot(fp, tp, label = "AUC = " + str(auc))
plt.title("ROC Curve - AdaBoost")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression

params = {"C": [0.1, 1, 2, 5, 10], "penalty": ["l1"], "solver": ["liblinear"]}
gs = GridSearchCV(estimator = LogisticRegression(), param_grid = params, scoring = "accuracy")
gs.fit(X_train, y_train)

y_test_ppred = gs.predict_proba(X_test)[:,1]
y_test_pred = [(x > 0.15) for x in y_test_ppred]

print(gs.best_params_)
print(gs.best_score_)

print(classification_report(y_test, y_test_pred))

fp, tp, t = roc_curve(y_test, y_test_ppred)
auc = roc_auc_score(y_test, y_test_ppred)
plt.plot(fp, tp, label = "AUC = " + str(auc))
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend()
plt.show()

model.coef_

pd.DataFrame({"Feature": X_train.columns, "Coef": [np.exp(x) for x in model.coef_[0]]})

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = Sequential()
model.add(Dense(50,input_shape=(66,), activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
ann = model.fit(x = X_train, y = y_train, verbose=0, epochs=50, validation_split=0.3, callbacks=[early_stop])

from sklearn.metrics import confusion_matrix

y_test_ppred = model.predict(X_test).ravel()
y_test_pred = [(x > 0.2) for x in y_test_ppred]
print(classification_report(y_test, y_test_pred))

print(confusion_matrix(y_test, y_test_pred))

fp, tp, t = roc_curve(y_test, y_test_ppred)
auc = roc_auc_score(y_test, y_test_ppred)
plt.plot(fp, tp, label = "AUC = " + str(auc))
plt.title("ROC Curve - ANN")
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend()
plt.show()

# fpl             -0.169515 -> poverty status
# 
# KHscore         -0.166111 -> financial knowledge score
# 
# PRODHAVE_1      -0.164885 -> Checking/savings at bank or credit union
# 
# PPINCIMP        -0.163251 -> household income
# 
# CONNECT         -0.145869 -> psychological connectedness
# 
# PRODHAVE_4      -0.145678 -> retirement account
# 
# LMscore         -0.139301 -> financial knowledge score
# 
# MANAGE1_1       -0.129274 -> paid all bills on time
# 
# ON2correct      -0.123166 -> objnumeracy2 answered correctly - “In the Bingo Lottery, the chance of winning a $10 prize is 1%. What is your best guess about how many people will win a $10 prize if 1,000 people each buy a single ticket for the Bingo Lottery?”
# 
# FWBscore        -0.122605 -> financial well-being score
# 
# KH2correct      -0.117137 -> KHKNOWL2 answered correctly - "Understanding of stocks vs bond vs savings volatility"
# 
# MATHARDSHIP_6    0.118326 -> utilities shut off due to non-payment
# 
# REJECTED_1       0.122898 -> applied for credit and was turned down
# 
# HOUSING          0.133260 -> "Which one of the following best describes your housing situation?"
# 
# MATHARDSHIP_3    0.136099 -> couldnt afford a place to live
# 
# FWB2_3           0.144405 -> "I am behind with my finances"
# 
# ENDSMEET         0.151467 -> Difficulty covering monthly bills and expenses
# 
# MATHARDSHIP_2    0.158996 -> Food didn't last and didn't have money to get more
# 
# MATHARDSHIP_1    0.177125 -> Worried whether food would run out, got money to buy more




