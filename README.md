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

8. # SVM
import pandas as pd

a = pd.read_csv("loan-test.csv")

X = a[["Gender", "Dependents", "Education", "Self_Employed",
       "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
       "Loan_Amount_Term", "Credit_History", "Property_Area"]]

X = X.fillna(X.mode().iloc[0])

y = a["Loan_status"]
y = y.fillna(y.mode()[0])

X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

st = StandardScaler()

X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

from sklearn.svm import SVC

sv = SVC(kernel="linear")

sv.fit(X_train, y_train)

pred = sv.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_test, pred))

print(confusion_matrix(y_test, pred))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
 # SVM Prediction
new = pd.DataFrame([{
    "Gender": 'Male',
    "Dependents": 0,
    "Education": 'Graduate',
    "Self_Employed": 'No',
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 'Urban'
}])
new = pd.get_dummies(new,drop_first=True)
new = new.reindex(columns=X.columns,fill_value=0)
new = st.fit_transform(new)
print(sv.predict(new))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
# Ridge Regression, Lasso Regression, Elastic Net Regression, evaluation
import pandas as pd
a=pd.read_csv("loan-test.csv")
X=a[["ApplicantIncome",	"CoapplicantIncome","LoanAmount"]]
X=X.fillna(X.mean())
y=a["Loan_Amount_Term"]
y=y.fillna(y.mode()[0])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

from sklearn.linear_model import ElasticNet
r = ElasticNet(alpha=1, l1_ratio=0.5)
r.fit(X_train,y_train)
pred = r.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
print(root_mean_squared_error(y_test,pred))
print(mean_squared_error(y_test,pred))
print(mean_absolute_error(y_test,pred))
print(r2_score(y_test,pred))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
# XGBClassifier
import pandas as pd

a=pd.read_csv("loan-test.csv")

X=a[["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]]

X=X.fillna(X.mean())

y=a["Loan_status"]

y=y.fillna(y.mode()[0])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y=lb.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred))
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
# K-Means
import pandas as pd
a=pd.read_csv("loan-test.csv")
X=a[["ApplicantIncome","CoapplicantIncome","LoanAmount","Credit_History"]]
X=X.fillna(X.median())

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
scaled = st.fit_transform(X)

from sklearn.cluster import KMeans
inertia = []
for i in range(1,11):
    m = KMeans(n_clusters=i)
    m.fit(scaled)
    inertia.append(m.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,11),inertia,marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

scores = []
from sklearn.metrics import silhouette_score
for j in range(2,9):
    s = KMeans(n_clusters=j,random_state=42,n_init=10)
    lable = s.fit_predict(scaled)
    score = silhouette_score(scaled,lable)
    scores.append(score)
    max(scores)
best_k = range(2,9)[scores.index(max(scores))]
best_k
    
model = KMeans(n_clusters=best_k,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=42)
label = model.fit_predict(scaled)
a["Cluster"]=lable
silhouette_score(scaled,lable)
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

# PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a=pd.read_csv("loan-test.csv")
X=a[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"]]
X=X.fillna(X.median())

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
scaled = st.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
sc = pca.fit_transform(scaled)

a["PCA1"]=sc[:,0]
a["PCA2"]=sc[:,1]

pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()

plt.scatter(a["PCA1"],a["PCA2"])
plt.show()
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
