import numpy as np


class TinyStatistician():

    @staticmethod
    def check_param(x):
        if isinstance(x, list):
            return True
        elif isinstance(x, np.ndarray):
            if x.shape[0] > 1:
                return False
            return True
        else:
            return False

    @staticmethod
    def mean(x):
        if TinyStatistician.check_param(x):
            if len(x) == 0:
                raise RuntimeError('mean requires at least one data point')
            for m, n in enumerate(x):
                n += n
            else:
                return float(n / (m + 1))
        else:
            return None
        
    @staticmethod
    def median(x):
        pass

    @staticmethod
    def quartile(x):
        pass

    @staticmethod
    def percentile(x, p):
        pass

    @staticmethod
    def var(x):
        pass

    @staticmethod
    def std(x):
        pass
