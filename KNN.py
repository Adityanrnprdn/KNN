import pandas as pd
from sklearn import datasets

#input & output sectioon

data = datasets.load_iris()
ip = data["data"]
ip = pd.DataFrame(data["data"])
ip.columns = ["SL","SW","PL","PW"]
op=pd.Series(data.target)

#training data

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

#import knn algorithum

from sklearn.neighbors import KNeighborsClassifier
alg=KNeighborsClassifier(n_neighbors=2)

alg.fit(xtr,ytr)

print("Accuracy: ",alg.score(xts,yts))

#prediction
yp=alg.predict(xts)
from sklearn import metrics
from sklearn.metrics import classification_report
classification_report(yts,yp)
cm=metrics.confusion_matrix(yts,yp)
print(cm)
import  numpy
ip=numpy.array([7.9	,3.8,6.4,2.0]).reshape(1,-1)
alg.predict(ip)


