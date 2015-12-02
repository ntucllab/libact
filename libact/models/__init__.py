"""
Concrete model classes.
"""
import logging

logger = logging.getLogger(__name__)

from .logistic_regression import LogisticRegression
from .perceptron import Perceptron
try:
    from .svm import SVM
except ImportError:
    logger.warn('libsvm python interface not found, SVM model not available.')
