def flatten_list(list):
    flat_list = [e for sl in list for e in sl]
    return flat_list

class Matrix():
    def __init__(self, val):
        if isinstance(val, list):
            if not all(isinstance(lst, list) for lst in val):
                raise TypeError("Sub-elements should be lists")
            if not all(isinstance(e, float) for lst in val for e in lst):
                raise TypeError("Sub-values of lists should be floats")
            self.data = val
            lst_column_len = [len(lst) for lst in val]
            first = lst_column_len[0]
            if all(e == first for e in lst_column_len):
                column_len = first
            else:
                raise ValueError("Sub-elements of Matrix must be of same size")
            self.shape = (len(val), column_len)
            if self.shape[0] < 2 or self.shape[1] < 2:
                raise AttributeError(f"Invalid shape: {self.shape}")
        elif isinstance(val, tuple):
            if not all(isinstance(num, int) for num in val):
                raise TypeError("Sub-elements should be int")                
            self.shape = val
            if self.shape[0] < 2 or self.shape[1] < 2:
                raise AttributeError(f"Invalid shape: {self.shape}")
            self.data = [[0 for e in range(0, val[1])] for lst in range(0, val[0])]
        else:
            raise TypeError(f"{type(val)} is not a supported type")
    
    def __add__(self, rhs):
        res = []
        for l1, l2 in zip(self.data, rhs.data):
            tmp_lst = []
            for e1, e2 in zip(l1, l2):
                tmp_lst.append(e1 + e2)
            res.append(tmp_lst)
        return Matrix(res)

    def __radd__(self, lhs):
        return lhs.__add__(self)
    
    def __sub__(self, rhs):
        res = []
        for l1, l2 in zip(self.data, rhs.data):
            tmp_lst = []
            for e1, e2 in zip(l1, l2):
                tmp_lst.append(e1 - e2)
            res.append(tmp_lst)
        return Matrix(res)
    
    def __rsub__(self, lhs):
        return lhs.__sub__(self)
    
    def __truediv__(self, divisor):
        if not isinstance(divisor, int):
            raise TypeError(f"{type(divisor)} not supported for division")
        if divisor == 0:
            raise ZeroDivisionError("Division by zero")
        res = []
        for lst in self.data:
            tmp_lst = []
            for e in lst:
                tmp_lst.append(e / divisor)
            res.append(tmp_lst)
        return Matrix(res)
    
    def __rtruediv__(self, divisor):
        return self.__truediv__(divisor)

    def compatible_dimension(self, m):
        return self.shape[1] == m.shape[0]

    def __mul__(self, rhs):
        if isinstance(rhs, int):
            res = []
            for lst in self.data:
                tmp_lst = []
                for e in lst:
                    tmp_lst.append(e * rhs)
                res.append(tmp_lst)
            return Matrix(res)
        elif isinstance(rhs, Matrix) and self.compatible_dimension(rhs):
            pass

    def __repr__(self):
        return f"Matrix({self.data}, {self.shape}"

    def __str__(self):
        ret = f"Matrix data: "
        ret += f"{self.data} and shape: {self.shape}"
        return ret


class Vector(Matrix):
    pass