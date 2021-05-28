import sklearn.metrics as metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def get_logistic_regression_acc(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)
    predicted = lr.predict(x_test)
    return round(metrics.accuracy_score(predicted, y_test),2)*100

def get_multinomial_naive_bayes_acc(x_train, y_train, x_test, y_test):
    mn_nb = MultinomialNB()
    mn_nb.fit(x_train, y_train)
    predicted = mn_nb.predict(x_test)
    return round(metrics.accuracy_score(predicted, y_test),2)*100
    
def get_linear_svm_acc(x_train, y_train, x_test, y_test):
    svm = LinearSVC(C=0.01)
    svm.fit(x_train, y_train)
    predicted = svm.predict(x_test)
    return round(metrics.accuracy_score(predicted, y_test),2)*100