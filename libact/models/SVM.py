from libact.base.interfaces import ContinuousModel
from svmutil import *

class SVM(ContinuousModel):


    def __init__(self, *args, **kwargs):
        self.svm_params = svm_parameter(args)

    def train(self, dataset, *args, **kwargs):
        self.model = svm_train(*dataset, self.svm_params)

    def predict(self, feature, *args, **kwargs):
        p_label, p_acc, p_val = svmpredict([], feature, self.model, args)
        return p_label

    def predict_real(self, feature, *args, **kwargs):
        params = '-b 1 ' + args
        p_label, p_acc, p_val = svmpredict([], feature, self.model, params)
        return p_val

    def score(self, testing_dataset, *args, **kwargs):
        p_label, p_acc, p_val = svmpredict([], feature, self.model)
        return p_acc

