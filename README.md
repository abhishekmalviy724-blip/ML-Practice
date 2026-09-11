# ML-Practice
1. # loan Amount Term Pridict
import pandas as pd
a = pd.read_csv("loan-test.csv")
X = a[["ApplicantIncome",
"CoapplicantIncome",
"Loan_Amount_Term"]]
X = X.fillna(X.mean())
y = a["LoanAmount"]
y = y.fillna(y.mean())
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y, test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(X_train,y_train)
y_pred = le.predict(X_test)
print(y_pred)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(le.coef_)
print(le.intercept_)
