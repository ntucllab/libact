from libact.base.interfaces import Model
import sklearn.linear_model
import svmutil

"""
A interface for libsvm's SVM model
"""
class SVM(Model):

    def __init__(self, *args, **kwargs):
        self.m = None

    def train(self, dataset, *args, **kwargs):
        X, y = zip(*dataset.get_labeled_entries())
        prob = svmutil.svm_problem([], y, [x.tolist() for x in X])
        param = svmutil.svm_parameter('-t 0 -c 0.1 -b 0 -q')
        self.m = svmutil.svm_train(prob, param)
        return self.m

    def predict(self, feature, *args, **kwargs):
        #TODO need only p_label
        p_label, p_acc, p_val = svmutil.svm_predict(y, feature, self.m)
        return p_label

    def score(self, testing_dataset, *args, **kwargs):
        if self.m == None:
            pass
        X, y = zip(*testing_dataset.get_labeled_entries())
        p_label, p_acc, p_val = svmutil.svm_predict(y, [x.tolist() for x in X],
                self.m)
        return p_acc[0] / 100. #ACC
