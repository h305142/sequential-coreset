import matplotlib.pyplot as plt
import numpy as np
import seaborn
import Utils


class GraphPlotter(object):
    def __init__(self, color_mathcing):
        """
        :param directoryName:  A directory that the graphs will be saved at
        """
        self.saveFigWidth = 20  # width of the figure
        self.saveFigHeight = 13  # height of the figure
        self.fontsize = 50  # font size of the letters at the axes
        self.legendfontsize = self.fontsize  # font size of the letters at the legend
        self.labelpad = 10  # padding with respect to the labels of the axes
        self.linewidth = 6  # line width of the graphs
        self.save_type = '.pdf'
        self.color_matching = color_mathcing  # dictionary of colors per each line

        self.OPEN_FIG = True  # automatically open figure after saving

        # updating plot parameters
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams['xtick.major.pad'] = '{}'.format(self.labelpad * 3)
        plt.rcParams['ytick.major.pad'] = '{}'.format(self.labelpad)
        plt.rcParams['xtick.labelsize'] = self.legendfontsize
        plt.rcParams['ytick.labelsize'] = self.legendfontsize
        seaborn.set_style("whitegrid")


