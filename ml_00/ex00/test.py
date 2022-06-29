import unittest
from matrix import Matrix
from matrix import Vector
from matrix import flatten_list
import numpy as np


class TestMatrixInit(unittest.TestCase):

    def setUp(self):
        self.lst_two_by_two = [[0.0, 1.0], [2.0, 3.0]]
        self.lst_two_by_three = [[0.0, 1.0, -1.0], [2.0, 3.0, -2.0]]
        self.lst_three_by_two = [[0.0, 1.0], [2.0, 3.0], [-1.0, -2.0]]
        self.shape_two_by_two = (2, 2)
        self.shape_two_by_three = (2, 3)
        self.lst_variable_len = [[0.0, 1.0], [2.0, 3.0, 42.0], [-1.0, -2.0]]
        self.lst2_col = [[5.0], [6.0], [7.0], [8.0], [9.0]]

    def test_init_matrix_from_two_by_two_lst(self):
        M1 = Matrix(self.lst_two_by_two)
        self.assertListEqual(M1.data, self.lst_two_by_two)
        self.assertEqual(M1.shape, (2, 2))

    def test_init_matrix_from_two_by_three_lst(self):
        M1 = Matrix(self.lst_two_by_three)
        self.assertListEqual(M1.data, self.lst_two_by_three)
        self.assertEqual(M1.shape, (2, 3))

    def test_init_matrix_from_three_by_two_lst(self):
        M1 = Matrix(self.lst_three_by_two)
        self.assertListEqual(M1.data, self.lst_three_by_two)
        self.assertEqual(M1.shape, (3, 2))

    def test_init_matrix_from_two_by_two_shape(self):
        M1 = Matrix(self.shape_two_by_two)
        flat_lst = flatten_list(M1.data)
        self.assertEqual(sum(flat_lst), 0)
        self.assertEqual(M1.shape, (2, 2))

    def test_init_matrix_from_two_by_three_shape(self):
        M1 = Matrix(self.shape_two_by_three)
        flat_lst = flatten_list(M1.data)
        self.assertEqual(sum(flat_lst), 0)
        self.assertEqual(M1.shape, (2, 3))

    def test_error_list_of_different_sizes(self):
        with self.assertRaises(ValueError) as e:
            expected = f"Sub-elements of Matrix must be of same size"
            M1 = Matrix(self.lst_variable_len)
        self.assertEqual(str(e.exception), expected)

    def test_type_error_invalid_init_val(self):
        with self.assertRaises(TypeError) as e:
            invalid_type = 42
            expected = f"{type(invalid_type)} is not a supported type"
            M1 = Matrix(invalid_type)
        self.assertEqual(str(e.exception), expected)

    def test_type_error_invalid_init_sub_list(self):
        with self.assertRaises(TypeError) as e:
            invalid_list_type = ['hello', 'world', '42']
            expected = "Sub-elements should be lists"
            M1 = Matrix(invalid_list_type)
        self.assertEqual(str(e.exception), expected)

    def test_type_error_invalid_init_internal_element(self):
        with self.assertRaises(TypeError) as e:
            invalid_list_type = [['hello', 'world'], ['42', '42']]
            expected = "Sub-values of lists should be floats"
            M1 = Matrix(invalid_list_type)
        self.assertEqual(str(e.exception), expected)

    def test_attribute_error_invalid_shape_from_list(self):
        with self.assertRaises(AttributeError) as e:
            expected = "Invalid shape: (5, 1)"
            M1 = Matrix(self.lst2_col)
        self.assertEqual(str(e.exception), expected)

    def test_attribute_error_invalid_shape_from_shape(self):
        with self.assertRaises(AttributeError) as e:
            expected = "Invalid shape: (5, 1)"
            M1 = Matrix((5, 1))
        self.assertEqual(str(e.exception), expected)

    def test_type_error_invalid_init_shape(self):
        with self.assertRaises(TypeError) as e:
            invalid_list_type = (2.0, 42.0)
            expected = "Sub-elements should be int"
            M1 = Matrix(invalid_list_type)
        self.assertEqual(str(e.exception), expected)


class TestMatrixCalculus(unittest.TestCase):
    def setUp(self):
        # Intialization
        self.lst1_two_by_two = [[0.0, 1.0], [2.0, 3.0]]
        self.lst2_two_by_two = [[1.0, 3.0], [-2.0, 5.0]]
        self.lst1_two_by_three = [[0.0, 1.0, -1.0], [2.0, 3.0, -2.0]]
        self.lst2_two_by_three = [[5.0, 4.0, -2.0], [2.0, -3.0, 4.0]]
        self.lst1_three_by_two = [[0.0, 1.0], [2.0, 3.0], [-1.0, -2.0]]
        self.lst2_three_by_two = [[0.0, -1.0], [-5.0, -3.0], [1.0, -2.0]]
        self.shape_two_by_two = (2, 2)
        self.shape_two_by_three = (2, 3)
        self.lst_variable_len = [[0.0, 1.0], [2.0, 3.0, 42.0], [-1.0, -2.0]]
        self.lst2_col = [[5.0], [6.0], [7.0], [8.0], [9.0]]
        # Matrice
        self.M1_two_by_two = Matrix(self.lst1_two_by_two)
        self.M2_two_by_two = Matrix(self.lst2_two_by_two)
        self.arr1_two_by_two = np.array(self.lst1_two_by_two)
        self.arr2_two_by_two = np.array(self.lst2_two_by_two)
        self.M1_two_by_three = Matrix(self.lst1_two_by_three)
        self.M2_two_by_three = Matrix(self.lst2_two_by_three)
        self.arr1_two_by_three = np.array(self.lst1_two_by_three)
        self.arr2_two_by_three = np.array(self.lst2_two_by_three)
        self.M1_three_by_two = Matrix(self.lst1_three_by_two)
        self.M2_three_by_two = Matrix(self.lst2_three_by_two)
        self.arr1_three_by_two = np.array(self.lst1_three_by_two)
        self.arr2_three_by_two = np.array(self.lst2_three_by_two)
        # Vectors
        self.lst1_col_three = [[0.0], [1.0], [2.0]]
        self.lst2_col_two = [[5.0], [6.0]]
        self.lst1_row_three = [[0.0, 1.0, 2.0]]
        self.lst2_row_two = [[5.0, 6.0]]
        self.v1_col_three = Vector(self.lst1_col_three)
        self.v2_col_two = Vector(self.lst2_col_two)
        self.v1_row_three = Vector(self.lst1_row_three)
        self.v2_row_two = Vector(self.lst2_row_two)

    def test_add_two_by_two_matrices(self):
        res = self.M1_two_by_two + self.M2_two_by_two
        expected = list(self.arr1_two_by_two + self.arr2_two_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_add_two_by_three_matrices(self):
        res = self.M1_two_by_three + self.M2_two_by_three
        expected = list(self.arr1_two_by_three + self.arr2_two_by_three)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_add_three_by_two_matrices(self):
        res = self.M1_three_by_two + self.M2_three_by_two
        expected = list(self.arr1_three_by_two + self.arr2_three_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_radd_two_by_two_matrices(self):
        res = self.M1_two_by_two.__radd__(self.M2_two_by_two)
        expected = list(self.arr2_two_by_two + self.arr1_two_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_radd_two_by_three_matrices(self):
        res = self.M1_two_by_three.__radd__(self.M2_two_by_three)
        expected = list(self.arr2_two_by_three + self.arr1_two_by_three)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_radd_three_by_two_matrices(self):
        res = self.M1_three_by_two.__radd__(self.M2_three_by_two)
        expected = list(self.arr2_three_by_two + self.arr1_three_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_sub_two_by_two_matrices(self):
        res = self.M1_two_by_two - self.M2_two_by_two
        expected = list(self.arr1_two_by_two - self.arr2_two_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_sub_two_by_three_matrices(self):
        res = self.M1_two_by_three - self.M2_two_by_three
        expected = list(self.arr1_two_by_three - self.arr2_two_by_three)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_sub_three_by_two_matrices(self):
        res = self.M1_three_by_two - self.M2_three_by_two
        expected = list(self.arr1_three_by_two - self.arr2_three_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rsub_two_by_two_matrices(self):
        res = self.M1_two_by_two.__rsub__(self.M2_two_by_two)
        expected = list(self.arr2_two_by_two - self.arr1_two_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rsub_two_by_three_matrices(self):
        res = self.M1_two_by_three.__rsub__(self.M2_two_by_three)
        expected = list(self.arr2_two_by_three - self.arr1_two_by_three)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rsub_three_by_two_matrices(self):
        res = self.M1_three_by_two.__rsub__(self.M2_three_by_two)
        expected = list(self.arr2_three_by_two - self.arr1_three_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_truediv_two_by_two_matrix(self):
        res = self.M1_two_by_two / 2
        expected = list(self.arr1_two_by_two / 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_truediv_two_by_three_matrix(self):
        res = self.M1_two_by_three / 2
        expected = list(self.arr1_two_by_three / 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_truediv_three_by_two_matrix(self):
        res = self.M1_three_by_two / 2
        expected = list(self.arr1_three_by_two / 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_truediv_unsupported_divisor(self):
        with self.assertRaises(TypeError) as e:
            invalid_divisor = 2.0
            expected = f"{type(invalid_divisor)} not supported for division"
            self.M1_two_by_three / invalid_divisor
        self.assertEqual(str(e.exception), expected)

    def test_truediv_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError) as e:
            invalid_divisor = 0
            expected = "Division by zero"
            self.M1_two_by_three / invalid_divisor
        self.assertEqual(str(e.exception), expected)

    def test_rtruediv_two_by_two_matrix(self):
        res = self.M1_two_by_two.__rtruediv__(2)
        expected = list(self.arr1_two_by_two / 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rtruediv_two_by_three_matrix(self):
        res = self.M1_two_by_three.__rtruediv__(2)
        expected = list(self.arr1_two_by_three / 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rtruediv_three_by_two_matrix(self):
        res = self.M1_three_by_two.__rtruediv__(2)
        expected = list(self.arr1_three_by_two / 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_two_by_two_matrix_by_scalar(self):
        res = self.M1_two_by_two * 2
        expected = list(self.arr1_two_by_two * 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_two_by_three_matrix_by_scalar(self):
        res = self.M1_two_by_three * -2
        expected = list(self.arr1_two_by_three * -2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_three_by_two_matrix_by_scalar(self):
        res = self.M1_three_by_two * 2
        expected = list(self.arr1_three_by_two * 2)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_two_by_two_matrices(self):
        res = self.M1_two_by_two * self.M2_two_by_two
        expected = list(np.dot(self.arr1_two_by_two, self.arr2_two_by_two))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_two_by_two_matrix_by_vector(self):
        res = self.M1_two_by_two * self.v2_col_two
        arr = np.array(self.lst2_col_two)
        expected = list(np.dot(self.arr1_two_by_two, arr))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_vector_by_two_by_two_matrix(self):
        res = self.v2_row_two * self.M1_two_by_two
        arr = np.array(self.lst2_row_two)
        expected = list(np.dot(arr, self.arr1_two_by_two))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_vector_by_three_by_two_matrix(self):
        res = self.v1_row_three * self.M1_three_by_two
        arr = np.array(self.lst1_row_three)
        expected = list(np.dot(arr, self.arr1_three_by_two))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_two_by_three_matrix_by_vector(self):
        res = self.M1_two_by_three * self.v1_col_three
        arr = np.array(self.lst1_col_three)
        expected = list(np.dot(self.arr1_two_by_three, arr))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_two_by_three_matrices(self):
        res = self.M1_two_by_three * self.M2_three_by_two
        expected = list(np.dot(self.arr1_two_by_three, self.arr2_three_by_two))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_mul_three_by_two_matrices(self):
        res = self.M1_three_by_two * self.M2_two_by_three
        expected = list(np.dot(self.arr1_three_by_two, self.arr2_two_by_three))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rmul_two_by_two_matrices(self):
        res = self.M1_two_by_two.__rmul__(self.M2_two_by_two)
        expected = list(np.dot(self.arr2_two_by_two, self.arr1_two_by_two))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rmul_two_by_three_matrices(self):
        res = self.M1_two_by_three.__rmul__(self.M2_three_by_two)
        expected = list(np.dot(self.arr2_three_by_two, self.arr1_two_by_three))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_rmul_three_by_two_matrices(self):
        res = self.M1_three_by_two.__rmul__(self.M2_two_by_three)
        expected = list(np.dot(self.arr2_two_by_three, self.arr1_three_by_two))
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_repr_two_by_two_matrix(self):
        res = self.M1_two_by_two
        expected = f"Matrix({self.lst1_two_by_two}, (2, 2))"
        self.assertEqual(res.__repr__(), expected)

    def test_str_two_by_two_matrix(self):
        res = self.M1_two_by_two
        expected = f"Matrix data: "
        expected += f"{self.lst1_two_by_two} and shape: (2, 2))"
        self.assertEqual(str(res), expected)

    def test_transpose_two_by_two_matrix(self):
        res = self.M1_two_by_two.T()
        expected = np.transpose(self.arr1_two_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))

    def test_transpose_two_by_three_matrix(self):
        res = self.M1_two_by_three.T()
        expected = np.transpose(self.arr1_two_by_three)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))
        self.assertEqual(res.shape, (3, 2))

    def test_transpose_three_by_two_matrix(self):
        res = self.M1_three_by_two.T()
        expected = np.transpose(self.arr1_three_by_two)
        self.assertListEqual(flatten_list(res.data), flatten_list(expected))
        self.assertEqual(res.shape, (2, 3))


class TestVectorInit(unittest.TestCase):
    def setUp(self):
        self.lst1_col = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        self.lst2_col = [[5.0], [6.0], [7.0], [8.0], [9.0]]
        self.lst1_row = [[0.0, 1.0, 2.0, 3.0, 4.0]]
        self.lst2_row = [[5.0, 6.0, 7.0, 8.0, 9.0]]
        self.v1_col = Vector(self.lst1_col)
        self.v2_col = Vector(self.lst2_col)
        self.v1_row = Vector(self.lst1_row)
        self.v2_row = Vector(self.lst2_row)
        self.arr1_col = np.array(self.lst1_col)
        self.arr2_col = np.array(self.lst2_col)
        self.arr1_row = np.array(self.lst1_row)
        self.arr2_row = np.array(self.lst2_row)
        self.s1 = 2

    # Testing vector initiation
    def test_init_vector_as_row(self):
        v1 = Vector(self.lst1_row)
        expected = np.array(self.lst1_row)
        expected = list(expected)
        self.assertListEqual(flatten_list(v1.data), flatten_list(expected))

    def test_init_vector_as_column(self):
        v1 = Vector(self.lst1_col)
        expected = list(np.array(self.lst1_col))
        self.assertListEqual(v1.data, list(expected))

    def test_init_vector_row_as_tuple(self):
        v1 = Vector((1, 5))
        expected = 0
        self.assertEqual(sum(flatten_list(v1.data)), expected)

    def test_init_vector_col_as_tuple(self):
        v1 = Vector((5, 1))
        expected = 0
        self.assertEqual(sum(flatten_list(v1.data)), expected)


class TestVectorOutput(unittest.TestCase):
    def setUp(self):
        self.lst1_col = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        self.lst2_col = [[5.0], [6.0], [7.0], [8.0], [9.0]]
        self.lst1_row = [[0.0, 1.0, 2.0, 3.0, 4.0]]
        self.lst2_row = [[5.0, 6.0, 7.0, 8.0, 9.0]]
        self.v1_col = Vector(self.lst1_col)
        self.v2_col = Vector(self.lst2_col)
        self.v1_row = Vector(self.lst1_row)
        self.v2_row = Vector(self.lst2_row)
        self.arr1_col = np.array(self.lst1_col)
        self.arr2_col = np.array(self.lst2_col)
        self.arr1_row = np.array(self.lst1_row)
        self.arr2_row = np.array(self.lst2_row)
        self.s1 = 2

    # Testing vector addition
    def test_mul_builtin_column_vectors(self):
        expected = np.multiply(self.arr1_col, self.s1)
        expected = list(expected)
        yours = self.v1_col.__mul__(self.s1)
        self.assertListEqual(flatten_list(yours.data), flatten_list(expected))

    def test_mul_builtin_row_vectors(self):
        expected = np.multiply(self.arr1_row, self.s1)
        expected = list(expected)
        yours = self.v1_row.__mul__(self.s1)
        self.assertListEqual(flatten_list(yours.data), flatten_list(expected))

    def test_mul_sign_builtin_column_vectors_by_scalar(self):
        expected = np.multiply(self.arr1_col, self.s1)
        expected = list(expected)
        yours = self.v1_col * self.s1
        self.assertListEqual(flatten_list(yours.data), flatten_list(expected))

    def test_mul_sign_builtin_row_vectors_by_scalar(self):
        expected = np.multiply(self.arr1_row, self.s1)
        expected = list(expected)
        yours = self.v1_row * self.s1
        self.assertListEqual(flatten_list(yours.data), flatten_list(expected))

    def test_rmul_builtin_column_vectors_by_scalar(self):
        expected = np.multiply(self.arr1_col, self.s1)
        expected = list(expected)
        yours = self.v1_col.__rmul__(self.s1)
        self.assertListEqual(flatten_list(yours.data), flatten_list(expected))

    def test_rmul_builtin_row_vectors_by_scalar(self):
        expected = np.multiply(self.arr1_row, self.s1)
        expected = list(expected)
        yours = self.v1_row.__rmul__(self.s1)
        self.assertListEqual(flatten_list(yours.data), flatten_list(expected))

    def test_str_magic_method(self):
        data_expected = [[0.0, 1.0, 2.0, 3.0, 4.0]]
        shape_expected = (1, 5)
        expected = f"Vector data: "
        expected += f"{data_expected} and shape: {shape_expected})"
        yours = self.v1_row.__str__()
        self.assertEqual(yours, expected)

    def test_str_ctor_magic_method(self):
        data_expected = [[0.0, 1.0, 2.0, 3.0, 4.0]]
        shape_expected = (1, 5)
        expected = f"Vector data: "
        expected += f"{data_expected} and shape: {shape_expected})"
        yours = str(self.v1_row)
        self.assertEqual(yours, expected)

    def test_repr_magic_method(self):
        data_expected = [[0.0, 1.0, 2.0, 3.0, 4.0]]
        shape_expected = (1, 5)
        expected = f"Vector({data_expected}, {shape_expected})"
        yours = self.v1_row.__repr__()
        self.assertEqual(yours, expected)

    def test_dot_row_x_col_vector(self):
        expected = np.dot(self.arr1_row, self.arr2_col)
        yours = self.v1_row.dot(self.v2_col)
        self.assertEqual(yours, expected)


# Cmd entrypoint for unittest
if __name__ == '__main__':
    unittest.main()
    # unittest.main(defaultTest='suite')
    # loader = unittest.TestLoader()
    # test_cases =
    # (TestMatrixInit, TestMatrixCalculus, TestVectorInit, TestVectorOutput)
    # suite = unittest.TestSuite()
    # for t in test_cases:
    # tests = loader.loadTestsFromTestCase(t)
    # suite.addTests(tests)
    # unittest.TextTestRunner(verbosity=2).run(suite)
