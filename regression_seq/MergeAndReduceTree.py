import numpy as np
import pandas as pd
from collections import deque
import Coreset


class MergeAndReduceTree(object):
    def __init__(self, file_path, leaf_size):
        self.file_path = file_path
        self.leaf_size = leaf_size
        self.tree = deque()

    def attainRoot(self):
        for data_chunk in pd.read_csv(self.file_path, chunksize=self.leaf_size):
            data_chunk = data_chunk.to_numpy()
            temp = Coreset.Coreset()

            self.tree.append(Coreset.Coreset())

