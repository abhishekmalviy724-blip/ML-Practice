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
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

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
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

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
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
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
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
5. # Tree Regration
6. import pandas as pd

a=pd.read_csv("loan-test.csv")

X=a[["ApplicantIncome", "CoapplicantIncome", "Credit_History"]]

y=a["LoanAmount"]

X=X.fillna(X.median())

y=y.fillna(y.median())

from sklearn.model_selection  import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeRegressor

dc=DecisionTreeRegressor(max_depth=5,random_state=42)

dc.fit(X_train,y_train)

pred=dc.predict(X_test)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

print(mean_absolute_error(y_test,pred))

print(r2_score(y_test,pred))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
6. # Random Forest
7. import pandas as pd
a=pd.read_csv("loan-test.csv")
X=a[["ApplicantIncome",
"CoapplicantIncome",
"LoanAmount",
"Credit_History"]]
X=X.fillna(X.median())
y=a["Loan_status"]
y=y.fillna(y.mode()[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rcf = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=10,min_samples_leaf=5,max_features='sqrt',criterion='entropy',random_state=42,oob_score=True)
rcf.fit(X_train,y_train)
pred = rcf.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

for feature, importance in zip(X.columns,rcf.feature_importances_):
    print(feature,importance)
    
print(rcf.predict_proba(new))
print(rcf.oob_score_)
    
new = [[6200, 2633, 150, 1]]
print(rcf.predict(new))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

7. # Bays Theorum
8. import pandas as pd
a=pd.read_csv("loan-test.csv")
X=a[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"]]
X=X.fillna(X.median())
y=a["Loan_status"]
y=y.fillna(y.mode()[0])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
pred=gnb.predict(X_test)
pred

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
pred2 = mnb.predict(X_test)
pred2

from sklearn.naive_bayes import BernoulliNB
bnb  = BernoulliNB()
bnb.fit(X_train,y_train)
pred3 = bnb.predict(X_test)
pred3

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,pred2))
print(confusion_matrix(y_test,pred2))
print(classification_report(y_test,pred2))

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,pred3))
print(confusion_matrix(y_test,pred3))
print(classification_report(y_test,pred3))

new = [[6200,2633,150,1]]
print(gnb.predict(new))
print(mnb.predict(new))
print(bnb.predict(new))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
