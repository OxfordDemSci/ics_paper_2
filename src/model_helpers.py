import pandas as pd
import statsmodels.api as sm

def run_OLS(df, dep_var, indep_vars):
    temp = df
    for var in indep_vars:
        temp = temp[temp[var].notnull()]
    temp = temp[ temp[dep_var].notnull()]
    y = temp[dep_var]
    X = temp[indep_vars]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())