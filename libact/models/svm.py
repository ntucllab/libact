from libact.base.interfaces import ContinuousModel
import svmutil

class SVM(ContinuousModel):


    def __init__(self, *args, **kwargs):
        self.svm_train_params = kwargs.pop('svm_train_params', '-q -b 1')
        self.svm_pred_params = kwargs.pop('svm_pred_params', '-q -b 1')

    def train(self, dataset, *args, **kwargs):
        X, y = zip(*dataset.get_labeled_entries())
        self.model = svmutil.svm_train(y, X, self.svm_train_params)

    def predict(self, feature, *args, **kwargs):
        svm_pred_params = kwargs.pop('svm_pred_params', self.svm_pred_params)
        rand_y = [-1] * len(feature)
        p_label, p_acc, p_val = svmutil.svm_predict(rand_y, feature, self.model, svm_pred_params)
        return p_label

    def predict_real(self, feature, *args, **kwargs):
        svm_pred_params = kwargs.pop('svm_pred_params', self.svm_pred_params)
        rand_y = [-1] * len(feature)
        p_label, p_acc, p_val = svmutil.svm_predict(rand_y, feature, self.model, svm_pred_params)
        return p_val

    def score(self, testing_dataset, *args, **kwargs):
        X, y = zip(*testing_dataset.get_labeled_entries())
        svm_pred_params = kwargs.pop('svm_pred_params', self.svm_pred_params)
        p_label, p_acc, p_val = svmutil.svm_predict(y, X, self.model, svm_pred_params)
        return p_acc[0]/100

