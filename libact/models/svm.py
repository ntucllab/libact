"""SVM

An interface for libsvm's SVM model. Please make sure libsvm is installed.
"""

import sklearn.linear_model
import svmutil

from libact.base.interfaces import Model


class SVM(Model):
    """Support Vector Machine Classifier

    Parameters
    ----------
    param : string, optional, default='-t 0 -c 0.1 -b 0 -q'
        | libsvm paramter options:
        | -s svm_type : set type of SVM (default 0)
        |     0 -- C-SVC
        |     1 -- nu-SVC
        |     2 -- one-class SVM
        |     3 -- epsilon-SVR
        |     4 -- nu-SVR
        | -t kernel_type : set type of kernel function (default 2)
        |     0 -- linear: u'*v
        |     1 -- polynomial: (gamma*u'*v + coef0)^degree
        |     2 -- radial basis function: exp(-gamma*|u-v|^2)
        |     3 -- sigmoid: tanh(gamma*u'*v + coef0)
        | -d degree : set degree in kernel function (default 3)
        | -g gamma : set gamma in kernel function (default 1/num_features)
        | -r coef0 : set coef0 in kernel function (default 0)
        | -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
        | (default 1)
        | -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
        | (default 0.5)
        | -p epsilon : set the epsilon in loss function of epsilon-SVR
        | (default 0.1)
        | -m cachesize : set cache memory size in MB (default 100)
        | -e epsilon : set tolerance of termination criterion (default 0.001)
        | -h shrinking: whether to use the shrinking heuristics, 0 or 1
        | (default 1)
        | -b probability_estimates: whether to train a SVC or SVR model for
        | probability estimates, 0 or 1 (default 0)
        | -wi weight: set the parameter C of class i to weight*C, for C-SVC
        | (default 1)

    Attributes
    ----------
    m : libsvm model instance
        Before training, m = None.
        After training, m = the return value from svmutil.svm_train.

    References
    ----------
    https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    """

    def __init__(self, *args, **kwargs):
        self.m = None

        param_str = kwargs.pop('param', '-t 0 -c 0.1 -b 0 -q')
        self.param = svmutil.svm_parameter(param_str)

    def train(self, dataset, *args, **kwargs):
        X, y = zip(*dataset.get_labeled_entries())
        prob = svmutil.svm_problem(y, [x.tolist() for x in X])
        self.m = svmutil.svm_train(prob, self.param)
        return self.m

    def predict(self, feature, *args, **kwargs):
        #TODO need only p_label
        p_label, p_acc, p_val = svmutil.svm_predict(None, feature, self.m)
        return p_label

    def score(self, testing_dataset, *args, **kwargs):
        if self.m == None:
            pass
        X, y = zip(*testing_dataset.get_labeled_entries())
        p_label, p_acc, p_val = svmutil.svm_predict(y, [x.tolist() for x in X],
                self.m)
        return p_acc[0] / 100. #ACC
