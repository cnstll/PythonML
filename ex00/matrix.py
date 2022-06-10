def flatten_list(list):
    flat_list = [e for sl in list for e in sl]
    return flat_list


class Matrix():
    def __init__(self, val):
        if isinstance(val, list):
            if len(val) == 0:
                raise ValueError("Empty list is not supported")
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
            if type(self) == Matrix:
                if self.shape[0] < 2 or self.shape[1] < 2:
                    raise AttributeError(f"Invalid shape: {self.shape}")
        elif isinstance(val, tuple):
            if not all(isinstance(num, int) for num in val):
                raise TypeError("Sub-elements should be int")
            self.shape = val
            if type(self) == Matrix:
                if self.shape[0] < 2 or self.shape[1] < 2:
                    raise AttributeError(f"Invalid shape: {self.shape}")
            nb_of_row = range(0, val[0])
            nb_of_col = range(0, val[1])
            self.data = [[0 for e in nb_of_col] for lst in nb_of_row]
        else:
            raise TypeError(f"{type(val)} is not a supported type")

    def is_col_vec(self, lst):
        if all(isinstance(sl, list) and len(sl) == 1 for sl in lst):
            return True
        else:
            return False

    def is_row_vec(self, lst):
        if len(lst) == 1:
            return True
        else:
            return False

    def __add__(self, rhs):
        if type(self) != Matrix or type(rhs) != Matrix:
            raise TypeError("Addition is only supported for matrices")
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
        if type(self) != Matrix or type(rhs) != Matrix:
            raise TypeError("Addition is only supported for matrices")
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
        if type(self) != Matrix:
            raise TypeError("Addition is only supported for matrices")
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
        if type(self) == Matrix:
            return Matrix(res)
        else:
            return Vector(res)

    def __rtruediv__(self, divisor):
        return self.__truediv__(divisor)

    def are_same_dimension(self, other):
        return self.shape == other.shape

    def compatible_dimension(self, m1, m2):
        return m1.shape[1] == m2.shape[0]

    def transpose(self, data):
        t_data = []
        sub_list = data[0]
        for col_num in range(0, len(sub_list)):
            row = [lst[col_num] for lst in data]
            t_data.append(row)
        return t_data

    def T(self):
        if type(self) != Matrix:
            raise TypeError("Tranpose is only handled for matrix type")
        data = self.transpose(self.data)
        return Matrix(data)

    def __mul__(self, rhs):
        res = []
        if isinstance(rhs, int):
            for lst in self.data:
                tmp_lst = []
                for e in lst:
                    tmp_lst.append(e * rhs)
                res.append(tmp_lst)
        elif isinstance(rhs, Matrix) and self.compatible_dimension(self, rhs):
            n_col = rhs.shape[1]
            n_row = self.shape[0]
            m = self.shape[1]
            res = [[0 for e in range(0, n_col)] for row in range(0, n_row)]
            for i in range(0, n_row):
                for j in range(0, n_col):
                    sum = 0
                    for k in range(0, m):
                        sum += self.data[i][k] * rhs.data[k][j]
                    res[i][j] = sum
        if type(self) == Matrix and type(rhs) in (Matrix, int):
            return Matrix(res)
        else:
            return Vector(res)

    def __rmul__(self, lhs):
        if isinstance(lhs, Matrix):
            return lhs.__mul__(self)
        else:
            return self.__mul__(lhs)

    def __repr__(self):
        if type(self) == Matrix:
            return f"Matrix({self.data}, {self.shape})"
        if type(self) == Vector:
            return f"Vector({self.data}, {self.shape})"

    def __str__(self):
        if type(self) == Matrix:
            ret = f"Matrix data: "
            ret += f"{self.data} and shape: {self.shape})"
            return ret
        if type(self) == Vector:
            ret = f"Vector data: "
            ret += f"{self.data} and shape: {self.shape})"
            return ret


class Vector(Matrix):
    def __init__(self, val):
        super().__init__(val)

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def dot(self, rhs):
        if type(rhs) != Vector:
            raise TypeError(f"Type {type(rhs)} not supported as operand")
        if super().compatible_dimension(self, rhs):
            row_flat = self.flatten_list(rhs.data)
            col_flat = self.flatten_list(self.data)
            res = sum(map(lambda a, b: a * b, row_flat, col_flat))
            return res
        else:
            raise ValueError("Invalid dot product")
