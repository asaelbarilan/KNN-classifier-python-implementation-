
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from scipy import stats



#seed
seed=3
np.random.seed(seed)

class kNNClassifier:
  def __init__(self, n_neighbors):
    self.n_neighbors = n_neighbors

  def fit(self, X, y):
    # your code goes here
    self.train_samples=X
    self.lables=y


  def predict(self, X):
    # your code goes here
    k=self.n_neighbors
    distance=np.zeros((X.shape[0],self.train_samples.shape[0]),dtype='float32')
    y_pred = np.zeros((X.shape[0]), dtype='float32')
    # for i,sample in enumerate(X):
    #     distance[i]=np.sqrt(np.sum(np.power(sample-X,2),axis=1))
    #     y_pred[i] = stats.mode(y_trained[list(np.argsort(distance[i])[1:k+1])])[0][0]
    for i,sample in enumerate(X):
        distance[i]=np.sqrt(np.sum(np.power(sample-self.train_samples,2),axis=1))
        y_pred[i] = stats.mode(self.lables[list(np.argsort(distance[i])[1:k+1])])[0][0]
    return y_pred



def true_boundary_voting_pred(wealth, religiousness):
    return religiousness -0.1*((wealth -5 )** 3 -wealth** 2 +(wealth -6 )** 2 +80)

def generate_data(m, seed=None):
    # if seed is not None, this function will always generate the same data
    X = np.random.uniform(low=0.0, high=10.0, size=(m ,2))
    y = np.sign(true_boundary_voting_pred(X[: ,0], X[: ,1]))
    y[y==0] = 1
    samples_to_flip = np.random.randint(0 , m//10)
    flip_ind = np.random.choice(m, samples_to_flip, replace=False)
    y[flip_ind] = -y[flip_ind]
    return X, y

def plot_labeled_data(ax, k, X, y, no_titles=False):
    republicans = (y == 1)
    democrats = (y == -1)
    ax.scatter(X[republicans, 0], X[republicans, 1], c='r')
    ax.scatter(X[democrats, 0], X[democrats, 1], c='b')
    ax.set_title('k: {}'.format(k))
    ax.plot(np.linspace(0, 10, 1000), -true_boundary_voting_pred(np.linspace(0, 10, 1000), np.zeros(1000)),
            linewidth=2, c='k')
def subplotlocation(i):
    if i<4:
        q=i
        p=0
    else:
        q=i-4
        p=1
    return p,q

def metricsMatrixs(y,y_pred,k,M,conf_mat,classification_metrics,accuracy_mat):
    M.loc[:, k] = y_pred
    conf_mat.loc[:, k] = sklearn.metrics.confusion_matrix(y, y_pred).flatten()
    classification_metrics[k] = sklearn.metrics.classification_report(y, y_pred)
    accuracy_mat.loc[:, k] = sklearn.metrics.accuracy_score(y, y_pred)
    return


Xtot,ytot=generate_data(m=1000)
X_train,y_train=Xtot[:600],ytot[:600]
X_val,y_val=Xtot[600:800],ytot[600:800]
X_test,y_test=Xtot[800:1000],ytot[800:1000]

k_list=[1, 3, 5, 11, 21, 51, 99]

accuracy_mat=pd.DataFrame(0, index=[0], columns=k_list)
conf_mat=pd.DataFrame(0,index=['tn', 'fp', 'fn', 'tp'], columns=k_list)
classification_metrics={}
#fig, axs = plt.subplots(4, 2)

kobject = kNNClassifier(5)
# we fit
kobject.fit(X_train,y_train)
#we train
y_pred=kobject.predict(X_train)
trainaccuracy=sklearn.metrics.accuracy_score(y_train, y_pred)

train_error=[]


X=X_train
y=y_train
M=pd.DataFrame(0, index=np.arange(len(X)), columns=k_list)
#we validate
for i,k in enumerate(k_list):
    kobject.n_neighbors=k
    y_pred=kobject.predict(X)
    #evaluate
    metricsMatrixs(y, y_pred, k, M, conf_mat, classification_metrics, accuracy_mat)
    #plotting
    # p,q=subplotlocation(i)
    # plot_labeled_data(axs[q, p], k, X, y_pred, no_titles=False)
    train_error.append(100*sum(y!=y_pred)/len(y))

plt.figure()
plt.plot(k_list,train_error,marker='o')
for i, txt in enumerate(k_list):
    plt.annotate(txt, (k_list[i], train_error[i]))
plt.xlabel('k')
plt.ylabel('error percentage %')
plt.title('train error')
plt.show()


k_optimal=accuracy_mat.iloc[0].idxmax()
msg='optimal k from train is :{}'.format(accuracy_mat.iloc[0].idxmax())
print(msg)

#plot_labeled_data(axs[3, 1], k, X, y_train, no_titles=False)
# for ax in fig.get_axes():
#     ax.label_outer()
# fig.suptitle("Train", fontsize=16)
# fig.show()


#validation
#fig_val, axs_val = plt.subplots(4, 2)
val_error=[]
X=X_val
y=y_val
M=pd.DataFrame(0, index=np.arange(len(X)), columns=k_list)
#we validate
for i,k in enumerate(k_list):
    kobject.n_neighbors=k
    y_pred=kobject.predict(X)
    #evaluate
    metricsMatrixs(y, y_pred, k, M, conf_mat, classification_metrics, accuracy_mat)
    #plotting
    # p,q=subplotlocation(i)
    # plot_labeled_data(axs_val[q, p], k, X, y_pred, no_titles=False)
    val_error.append(100 * sum(y != y_pred) / len(y))

plt.figure()
plt.plot(k_list, val_error, marker='o')
for i, txt in enumerate(k_list):
    plt.annotate(txt, (k_list[i], val_error[i]))
plt.xlabel('k')
plt.ylabel('error percentage %')
plt.title('val error')
plt.show()

#plot_labeled_data(axs_val[3, 1], k, X, y_val, no_titles=False)
# for ax in fig_val.get_axes():
#     ax.label_outer()
msg='optimal k from val is :{}'.format(accuracy_mat.iloc[0].idxmax())
print(msg)
# fig_val.suptitle("validation", fontsize=16)
# fig_val.show()

#test
#fig_test, axs_test = plt.subplots(4, 2)
test_error=[]
X=X_test
y=y_test
M=pd.DataFrame(0, index=np.arange(len(X)), columns=k_list)
#we validate
for i,k in enumerate(k_list):
    kobject.n_neighbors=k
    y_pred=kobject.predict(X)
    #evaluate
    metricsMatrixs(y, y_pred, k, M, conf_mat, classification_metrics, accuracy_mat)
    #plotting
    #p,q=subplotlocation(i)
    #plot_labeled_data(axs_test[q, p], k, X, y_pred, no_titles=False)
    test_error.append(100 * sum(y != y_pred) / len(y))

plt.figure()
plt.plot(k_list, test_error, marker='o')
for i, txt in enumerate(k_list):
    plt.annotate(txt, (k_list[i], test_error[i]))
plt.xlabel('k')
plt.ylabel('error percentage %')
plt.title('test error')
plt.show()



msg='optimal k from test is :{}'.format(accuracy_mat.iloc[0].idxmax())
print(msg)

# plot_labeled_data(axs_test[3, 1], k, X, y_test, no_titles=False)
# for ax in fig_test.get_axes():
#     ax.label_outer()
# fig_test.suptitle("test", fontsize=16)
# fig_test.show()

###cross_validation###

import numpy as np
from sklearn.model_selection import KFold

k_list=[1, 3, 5, 11, 21, 51, 99]
kf = KFold(n_splits=5)
fig_cv, axs_cv = plt.subplots(4, 2)
z=1
for train_index, test_index in kf.split(X_test):

    X = X_test[test_index]
    y = y_test[test_index]
    kobject = kNNClassifier(5)
    # we fit for cv train
    kobject.fit(X, y)
    M=pd.DataFrame(0, index=np.arange(len(X)), columns=k_list)
    #we validate for different k's for test data
    for i,k in enumerate(k_list):
        kobject.n_neighbors=k
        y_pred=kobject.predict(X)
        accuracy_mat.loc[:, k] = sklearn.metrics.accuracy_score(y, y_pred)
        #evaluate
        #metricsMatrixs(y, y_pred, k, M, conf_mat, classification_metrics, accuracy_mat)
        #plotting
        #lot_labeled_data(axs_cv[q, p], k, X, y_pred, no_titles=False)
    msg='optimal k from kfold {} cv is :{}'.format(z,accuracy_mat.iloc[0].idxmax())
    print(msg)
    # plot_labeled_data(axs_cv[3, 1], 'real data', X, y_test, no_titles=False)
    # for ax in fig_test.get_axes():
    #     ax.label_outer()
    # fig_cv.suptitle("test", fontsize=16)
    # fig_cv.show()
    z+=1
