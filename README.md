# ML-Practice
1. # loan Amount Term Pridict
import pandas as pd
a = pd.read_csv("loan-test.csv")
X = a[["ApplicantIncome","CoapplicantIncome","LoanAmount"]]
X = X.fillna(X.mean())
y = a["Loan_Amount_Term"]
y = y.fillna(y.mean())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

from sklearn.preprocessing import PolynomialFeatures
ploy = PolynomialFeatures(degree=2)
x_ploy = ploy.fit_transform(X_train)
x_ploy = ploy.transform(X_test)
print(x_ploy)

from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(X_train,y_train)
y_pred = le.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_absolute_error(y_test,y_pred)
mae = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(le.coef_)
print(le.intercept_)
____________________________________________________________________________________________________________________________________________________________________________

2. # Logistic Regaression
import pandas as pd
a = pd.read_csv("loan-test.csv")
X = a[["ApplicantIncome","CoapplicantIncome","LoanAmount"]]
X = X.fillna(X.mean())
y = a["Self_Employed"]
y = y.fillna(y.mode()[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
____________________________________________________________________________________________________________________________________________________________________________

3. # Knn 
import pandas as pd

a = pd.read_csv("loan-test.csv")

X=  a[["ApplicantIncome",
"CoapplicantIncome",
"LoanAmount"]]

X = X.fillna(X.mean())

y=a["Self_Employed"]

y=y.fillna(y.mode()[0])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

b=le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
r=st.fit_transform(X_train)
t=st.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=5)
kn.fit(r,y_train)
y_pre = kn.predict(t)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pre))
print(confusion_matrix(y_test,y_pre))
____________________________________________________________________________________________________________________________________________________________________________
4. # Tree
5. import pandas as pd
a = pd.read_csv("loan-test.csv")
X=a[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]]
y=a["Loan_status"]
X=X.fillna(X.mean())
y=y.fillna(y.mode()[0])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(max_depth=3)
X_train_d=dc.fit(X_train,y_train)
pred = dc.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))
____________________________________________________________________________________________________________________________________________________________________________
