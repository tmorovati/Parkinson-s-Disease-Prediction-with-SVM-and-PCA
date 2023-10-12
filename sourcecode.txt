#read csv file which contains the dataset train data and test data
import pandas as pd
dataset = pd.read_csv("D://My Documents//ML_HW//ADproject//ReplicatedAcousticFeatures-ParkinsonDatabase.csv")

#if u wanna see how your dataset looks  like
print(dataset.head())
#print(dataset.shape())


#reading features
for row in dataset:
    print(row)
#asign features to X
X = dataset.drop('Status' , 1)
#asign lables to y
y= dataset['Status']

#print (y)
print("space")
#print(X)

#divide the entire data into 2 part
#train data : data we use to train out machine
#test data : data we use to Evaluate the Algorithm

from sklearn.model_selection import train_test_split
#train_test_split function parameters :https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.37)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X, y, test_size=0.2, random_state=0)

#checking the number of test and train data  each time
print (X_train)
print("df")
print (X_test)

#normalize feature set to perform PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_pca = sc.fit_transform(X_train_pca)
X_test_pca = sc.transform(X_test_pca)

#apply PCA
from sklearn.decomposition import PCA
#create a object of PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_train_pca)


explained_variance = pca.explained_variance_ratio_
print(explained_variance)
first_pca = max(explained_variance)
print(first_pca)


#train the SVM on traing data
from sklearn.svm import SVC
#SVC takes 2 parameters
#SVC function parameter : https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/svm/classes.py#L429
svclassifier = SVC(kernel='linear' ,gamma  = 'auto')
#fit method of SVC class is called to train the algorithm on the training data
#fit function parameters : https://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html
svclassifier.fit(X_train, y_train)

#make predictions
y_pred = svclassifier.predict(X_test)
print(y_pred)

#finding out the accuracy
from sklearn import metrics
#accuracy_score function in metrics lib parametes :https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
print("Accuracy_linear:",metrics.accuracy_score(y_test, y_pred ,True))
#the third argument is True by default if u change it to the False it returns the number of correctly classified samples
print("number of the correctly classified samples_linear:",metrics.accuracy_score(y_test, y_pred , False ))

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC

import seaborn as sns; sns.set()
# figure number
fignum = 1

# fit the model
for name, penalty in (('unreg', 1), ('reg', 0.05)):

    clf = SVC(kernel='linear', C=penalty)
    clf.fit(X, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()
'''


#implement polynomial kernel
x_poly_kernel = X
y_poly_kernel = y

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X, y, test_size =0.4)


svclassifier_poly = SVC(kernel='poly' , degree=8 ,gamma = 'auto' )
svclassifier_poly.fit(X_train_poly, y_train_poly)

y_predic_poly = svclassifier_poly.predict(X_test_poly)
print(y_predic_poly)


print("Accuracy_polynomial:",metrics.accuracy_score(y_test_poly, y_predic_poly ,True))
print("number of the correctly classified samples_polynomial:",metrics.accuracy_score(y_test_poly, y_predic_poly , False ))



#implementing Guassian kernel

X_train_guas, X_test_guas, y_train_guas, y_test_guas = train_test_split(X, y, test_size =0.45)

svclassifier_guas = SVC(kernel='rbf' , gamma= 'auto')
svclassifier_guas.fit(X_train_guas, y_train_guas)

y_predic_guas= svclassifier_guas.predict(X_test_guas)
print(y_predic_guas)

print("Accuracy_guassian:",metrics.accuracy_score(y_test_guas, y_predic_guas ,True))
print("number of the correctly classified samples_guassian:",metrics.accuracy_score(y_test_guas, y_predic_guas , False ))



#implement sigmoid kernel

X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(X, y, test_size =0.15)

svclassifier_sig = SVC(kernel='rbf' , gamma= 'auto')
svclassifier_sig.fit(X_train_sig, y_train_sig)

y_predic_sig= svclassifier_sig.predict(X_test_sig)
print(y_predic_sig)

print("Accuracy_sigmoid:",metrics.accuracy_score(y_test_sig, y_predic_sig))
print("number of the correctly classified samples_sigmoid:",metrics.accuracy_score(y_test_sig, y_predic_sig , False ))






