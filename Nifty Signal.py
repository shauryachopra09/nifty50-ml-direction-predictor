import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
nifty50=yf.Ticker("^NSEI")
nifty50=nifty50.history(period="max")
print(nifty50)
print(nifty50.index)
nifty50.plot.line(y="Close", use_index=True)
plt.show()
del nifty50["Dividends"]
del nifty50["Stock Splits"]
nifty50["Tomorrow"]=nifty50["Close"].shift(-1)
print(nifty50)
nifty50["Target"]=(nifty50["Tomorrow"]>nifty50["Close"]).astype(int)
print(nifty50)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train=nifty50.iloc[:-100]
test=nifty50.iloc[-100:]
predictors=["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score
preds=model.predict(test[predictors])
preds=pd.Series(preds, index=test.index)
print(preds)
print(precision_score(test["Target"], preds))

combined=pd.concat([test["Target"], preds],axis=1)
combined.plot()
plt.show()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds=model.predict(test[predictors])
    preds=pd.Series(preds, index=test.index, name="Predictions")
    combined=pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step= 250):
    all_predictions=[]
    for i in range(start, data.shape[0], step):
        train=data.iloc[0:i]
        test=data.iloc[i:(i+step)]
        predictions=predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions=backtest(nifty50, model, predictors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts()/predictions.shape[0])

horizons=[2,5,60,250,1000]
new_predictors=[]
for horizon in horizons:
    rolling_averages=nifty50.rolling(horizon).mean()

    ratio_column=f"Close_Ratio_{horizon}"
    nifty50[ratio_column]=nifty50["Close"]/rolling_averages["Close"]
    trend_column=f"Trend_{horizon}"
    nifty50[trend_column]=nifty50.shift(1).rolling(horizon).sum()["Target"]
    new_predictors+=[ratio_column, trend_column]

nifty50=nifty50.dropna()
print(nifty50)

model=RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds>=.6]=1
    preds[preds < .6] = 0
    preds=pd.Series(preds, index=test.index, name="Predictions")
    combined=pd.concat([test["Target"], preds], axis=1)
    return combined

predictions=backtest(nifty50, model, new_predictors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))

# Price Movement Prediction Using Random Forest (Nifty 50, 2007–2025)
# Built a classification model using trend and ratio-based features to
# predict next-day index direction; achieved 57.6% precision on positive
# signals over 3,000+ data points using rolling-window backtesting.