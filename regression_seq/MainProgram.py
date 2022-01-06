import numpy as np
from scipy import stats
import Utils
import Optimizor
from multiprocessing.pool import ThreadPool
import Coreset
import copy


class MainProgram(object):
    def __init__(self, dataset, problem_type, Z, LAMBDA=1, streaming=False):

        self.pool = ThreadPool(Utils.THREAD_NUM)
        var_dict = Utils.initializaVariables(problem_type, Z, LAMBDA)
        self.coresets = [Coreset.Coreset(P=None, W=None, _sense_bound_lambda=var_dict['SENSE_BOUND'],
                                         max_ellipsoid_iters=var_dict['ELLIPSOID_MAX_ITER'],
                                         problem_type=var_dict['PROBLEM_TYPE'], use_svd=var_dict['USE_SVD'])
                         for i in range(Utils.REPS)]
        self.samplingProcedures = \
            (lambda i, sensitivity, sample_size, random_state=None:
             self.coresets[i].sampleCoreset(P=self.P, sensitivity=sensitivity,
                                            sample_size=sample_size, random_state=random_state)) if not streaming else 1

        self.P = Utils.readRealData(dataset)
        self.sample_sizes = Utils.generateSampleSizes(self.P.n)
        self.optimizor = Optimizor.Optimizor(P=self.P, Z=var_dict['Z'], C=var_dict['LAMBDA'],
                                             problem_type=var_dict['PROBLEM_TYPE'], tol=var_dict['OPTIMIZATION_TOL'],
                                             objective_cost=var_dict['OBJECTIVE_COST'])
        self.optimizor.defineSumOfWegiths(self.P.W)
        if not streaming:
            self.sensitivity = self.coresets[0].computeSensitivity(self.P)

        # self.opt_val, self.opt_time = self.optimizor.fit(self.P)

    def construct_coreset(self, sample_size = 200):
        coresets = [self.samplingProcedures(0, self.sensitivity, sample_size)]
        coreset = coresets[0]
        return coreset


    @staticmethod
    def main(trainfile, type = 'svm',sample_size = 200,streaming=False): # logisitic, nclz, svm, restricted_lz, lz, lse
        #dataset = 'HTRU_2.csv'
        #problem_type = 'svm'
        dataset = trainfile
        problem_type = type
        Z = 2
        Lambda = 1
        main_runner = MainProgram(dataset, problem_type, Z, Lambda, streaming)
        return main_runner.construct_coreset(sample_size)

# if __name__ == '__main__':
#     MainProgram.main()