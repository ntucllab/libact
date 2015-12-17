"""
Concrete model classes.
"""
import logging

logger = logging.getLogger(__name__)

from .logistic_regression import LogisticRegression
from .perceptron import Perceptron
from .svm import SVM
