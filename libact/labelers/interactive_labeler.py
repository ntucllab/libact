"""Interactive Labeler

This module includes an InteractiveLabeler.
"""
import matplotlib.pyplot as plt

from libact.base.interfaces import Labeler


class InteractiveLabeler(Labeler):
    """Interactive Labeler

    InteractiveLabeler is a Labeler object that shows the feature through image
    using matplotlib and lets human label each feature through command line
    interface.

    Parameters
    ----------
    labels: list
        List of valid labels.

    """

    def __init__(self, **kwargs):
        self.lbls = kwargs.pop('labels', None)

    def label(self, feature):
        plt.imshow(feature, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.draw()

        banner = "Enter the associated label with the image: "

        if self.lbls is not None:
            banner += str(self.lbls) + ' '
        lbl = input(banner)

        if (self.lbls is not None) and (int(lbl) not in self.lbls):
            raise ValueError('Invalid label.')

        return int(lbl)
