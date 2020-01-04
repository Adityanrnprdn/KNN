import pandas as pd
from sklearn import datasets

#input & output section
data=datasets.load_iris()
ip=data["data"]
ip=pd.DataFrame(data["data"]) 
ip.columns = ["SL","SW","PL","PW"]
op=pd.Series(data.target)

#training data

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.2)

num_classes = op.unique().shape[0]
from sklearn.cluster import KMeans
alg = KMeans(n_clusters=num_classes)
#prediction
alg.fit(xtr)
k=alg.predict(ip)
print(k)

#to check accuracy
print("Accuracy: ",abs(alg.score(ip,op)))

#plotting the clusters
print("Center: ",alg.cluster_centers_) #for finding center of cluster
x=ip["SL"]
y=ip["PL"]

import matplotlib.pyplot as plt
plt.scatter(x,y,c=k)
plt.scatter(alg.cluster_centers_[:,0],
           alg.cluster_centers_[:,2], c = "k",
            marker = "*",s = 100)
           
plt.show()


#PREDICTION
import pandas as pd
import numpy as np
from sklearn import svm,datasets
import matplotlib

iris = datasets.load_iris()
X = iris.data[:, :2]#we only take the first tw features'
y=iris.target

#plot resulting Support Vector boundaries with original data
#create fake input data for prediction that we will use for lotting
#create a meesh to plot in

x_min, x_max = X[:,0].min() -1,X[:,0].max() +1
y_min, y_max=X[:,1].min() -1,X[:,1].max()+1
h = (x_max/x_min)/100
xx,yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
X_plot = np.c_[xx.ravel(),yy.ravel()]

#create the SVC model object
C = 300.0 #SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C,gamma=0.08)
svc.fit(X,y)
Z = svc.predict(X_plot)
Z= Z.reshape(xx.shape)

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.contourf(xx,yy,Z,cmap=plt.cm.tab10,alpha=0.9)
plt.scatter(X[:, 0],X[:, 1],c=y,cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(),xx.max())
plt.title('SVC with linear kernel')