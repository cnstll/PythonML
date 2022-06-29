import math
import numpy as np


class TinyStatistician():

    @staticmethod
    def check_param(x):
        if isinstance(x, list):
            print(len(x))
            if len(x) == 0:
                return False
            return True
        elif isinstance(x, np.ndarray):
            if len(x) == 0 or (len(x.shape) == 2 and x.shape[1] > 1):
                return False
            return True
        else:
            return False

    @staticmethod
    def check_percentile(p):
        return p in range(1, 101) and isinstance(p, int)

    @staticmethod
    def mean(x):
        if TinyStatistician.check_param(x):
            sum = 0
            for m, n in enumerate(x):
                sum += n
            else:
                return float(sum / (m + 1))
        else:
            return None

    @staticmethod
    def median(x):
        return TinyStatistician.percentile(x, 50)

    @staticmethod
    def quartile(x):
        first_qrt = TinyStatistician.percentile(x, 25)
        third_qrt = TinyStatistician.percentile(x, 75)
        if first_qrt is None or third_qrt is None:
            return None
        else:
            return (first_qrt, third_qrt)

    @staticmethod
    def percentile(x, p):
        if TinyStatistician.check_param(x) \
           and TinyStatistician.check_percentile(p):
            x = sorted(x)
            p = p / 100
            rank = p * (len(x) - 1) + 1
            lo_rank = math.floor(rank)
            hi_rank = math.ceil(rank)
            lo_value = x[lo_rank - 1]
            hi_value = x[hi_rank - 1]
            diff = hi_value - lo_value
            mod = rank % math.floor(rank)
            interpolated_perc = lo_value + mod * diff
            return interpolated_perc
        else:
            return None

    @staticmethod
    def var(x):
        if TinyStatistician.check_param(x):
            avg = TinyStatistician.mean(x)
            print(avg)
            sum = 0
            for m, n in enumerate(x):
                sum += pow(n - avg, 2)
                print(n, avg)
            else:
                return round(float(sum / m), 1)
        else:
            return None

    @staticmethod
    def std(x):
        if TinyStatistician.check_param(x):
            return math.sqrt(TinyStatistician.var(x))
        else:
            return None
