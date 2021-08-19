import tkinter as tk
import cv2
import numpy as np
import math

from PIL import Image
from math import pi, sin, cos


# TODO RENAME TO POINT, mb change variables in out
class KnownPoint:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class Camera:
    def __init__(self, x, y, z, f_x, f_y, s_x, s_y, target_x, target_y):
        # TODO know about point C
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        # set starting points
        self.f_x = float(f_x)
        self.f_y = float(f_y)
        self.s_x = float(s_x)
        self.s_y = float(s_y)

        # point to find
        self.target_x = float(target_x)
        self.target_y = float(target_y)

        self.vector_to_target = [0, 0, 0]

        # TODO: DELETE
        self.vector_to_targetD = [0, 0, 0]

    def find_point_on_camera(self, f_starting_point: KnownPoint, s_starting_point: KnownPoint):

        # rotate_angle = self.f_y / (math.sqrt(self.f_x ** 2 + self.f_y ** 2))
        # f_starting_point_on_camera = [self.f_x, self.f_y]
        # s_starting_point_on_camera = [self.s_x, self.s_y]

        # TODO: write rotate of vector_cf(and vector cs)
        # f_starting_point_on_camera_rotate = rotate_vector(f_starting_point_on_camera, rotate_angle)
        # s_starting_point_on_camera_rotate = rotate_vector(s_starting_point_on_camera, rotate_angle)

        vector_cf = [f_starting_point.x - self.x, f_starting_point.y - self.y, f_starting_point.z - self.z]
        vector_cs = [s_starting_point.x - self.x, s_starting_point.y - self.y, s_starting_point.z - self.z]
        stretch_ratio_of_cf = []
        stretch_ratio_of_cs = []
        square_of_base_length = []
        stretch_ratio_of_cf, stretch_ratio_of_cs, square_of_base_length = self.find_stretch_ratio_and_pixel_size \
            (f_starting_point, s_starting_point)

        # TODO: DELETE!!
        vector_coD = []
        z_roots = []
        many_solutions = 0
        x, y, z = 0, 0, 0

        # TODO check second solution of m
        vector_co = []
        

        # print("stretch_ratio_of_cf "+str(stretch_ratio_of_cf))
        # print("stretch_ratio_of_cs " + str(stretch_ratio_of_cs))
        # print("m" + str(math.sqrt(square_of_base_length)))
        # print(vector_co)
        # print(vector_coD)
        self.vector_to_target = self.count_target(vector_co, vector_cf, vector_cs, stretch_ratio_of_cf,
                                                  stretch_ratio_of_cs, square_of_base_length)

        vector_to_targ_rotate = self.find_target(vector_co, vector_cf, stretch_ratio_of_cf, square_of_base_length)

        self.vector_to_targetD = self.count_target(vector_coD, vector_cf, vector_cs, stretch_ratio_of_cf,
                                                   stretch_ratio_of_cs, square_of_base_length)

        vector_to_targD_rotate = self.find_target(vector_coD, vector_cf, stretch_ratio_of_cf, square_of_base_length)

        print("vector to target")
        print(self.vector_to_target)
        print("vector to target through roated")
        print(vector_to_targ_rotate)
        print("vector to targetD")
        print(self.vector_to_targetD)
        print("vector to targetD through roated")
        print(vector_to_targD_rotate)
        print("vector cf")
        print(multiply_vector_on_scalar(vector_cf, stretch_ratio_of_cf))

        print("\n\n\n\n\n")

    def get_coordinates(self):
        return [self.x, self.y, self.z]

    def find_stretch_ratio_and_pixel_size(self, f_starting_point: KnownPoint, s_starting_point: KnownPoint):
        vector_cf = [f_starting_point.x - self.x, f_starting_point.y - self.y, f_starting_point.z - self.z]
        vector_cs = [s_starting_point.x - self.x, s_starting_point.y - self.y, s_starting_point.z - self.z]
        if abs(vector_cs[0]) < 0.000001 and abs(vector_cs[1]) < 0.000001 and abs(vector_cs[2]) < 0.000001:
            assert 0
        if abs(vector_cf[0]) < 0.000001 and abs(vector_cf[1]) < 0.000001 and abs(vector_cf[2]) < 0.000001:
            assert 0

        s_coeff = scalar_product(vector_cf, vector_cf) * scalar_product(vector_cs, vector_cs) * \
                  (self.f_x * self.s_x + self.f_y * self.s_y) ** 2 - (self.f_x ** 2 + self.f_y ** 2) * (
                          self.s_x ** 2 + self.s_y ** 2) * (scalar_product(vector_cf, vector_cs) ** 2)
        m_coeff = 2 * (self.f_x * self.s_x + self.f_y * self.s_y) * scalar_product(vector_cf, vector_cf) * \
                  scalar_product(vector_cs, vector_cs) - \
                  (self.f_x ** 2 + self.f_y ** 2 + self.s_x ** 2 + self.s_y ** 2) * \
                  scalar_product(vector_cf, vector_cs) ** 2
        j_coeff = scalar_product(vector_cs, vector_cs) * scalar_product(vector_cf, vector_cf) - \
                  scalar_product(vector_cs, vector_cf) ** 2

        square_of_base_length = solve_quadratic_equation(s_coeff, m_coeff, j_coeff)
        # find square of length
        #square_of_base_length = max(square_of_base_length)
        #if square_of_base_length <= 0:
        #    assert 0

        # base_length = math.sqrt(square_of_base_length)

        #stretch_ratio_of_cf = math.sqrt(((self.f_x ** 2 + self.f_y ** 2) * square_of_base_length + 1) /
        #                                (scalar_product(vector_cf, vector_cf)))
        #stretch_ratio_of_cs = math.sqrt(((self.s_x ** 2 + self.s_y ** 2) * square_of_base_length + 1) /
        #                                (scalar_product(vector_cs, vector_cs)))
        #return stretch_ratio_of_cf, stretch_ratio_of_cs, square_of_base_length

        stretch_ratio_of_cf = [math.sqrt(((self.f_x ** 2 + self.f_y ** 2) * square_of_base_length[0] + 1) /
                                        (scalar_product(vector_cf, vector_cf))),
                               math.sqrt(((self.f_x ** 2 + self.f_y ** 2) * square_of_base_length[1] + 1) /
                                         (scalar_product(vector_cf, vector_cf)))]

        stretch_ratio_of_cs = [math.sqrt(((self.s_x ** 2 + self.s_y ** 2) * square_of_base_length[0] + 1) /
                                        (scalar_product(vector_cs, vector_cs))),
                               math.sqrt(((self.s_x ** 2 + self.s_y ** 2) * square_of_base_length[1] + 1) /
                                        (scalar_product(vector_cs, vector_cs)))]

        return stretch_ratio_of_cf, stretch_ratio_of_cs, square_of_base_length

    def find_oy(self, vector_co, vector_cf, vector_cs, stretch_ratio_of_cf, stretch_ratio_of_cs, square_of_base_length):
        radius_vector_to_f_on_camera = sum_of_vectors(minus_vector(vector_co),
                                                      multiply_vector_on_scalar(vector_cf, stretch_ratio_of_cf))
        radius_vector_to_s_on_camera = sum_of_vectors(minus_vector(vector_co),
                                                      multiply_vector_on_scalar(vector_cs, stretch_ratio_of_cs))
        vector_ox_direction = sum_of_vectors(multiply_vector_on_scalar(radius_vector_to_s_on_camera, self.f_x),
                                             minus_vector(multiply_vector_on_scalar(radius_vector_to_f_on_camera,
                                                                                    self.s_x)))
        return multiply_vector_on_scalar(vector_ox_direction, 1 / (self.s_y * self.f_x - self.s_x * self.f_y))

    # @staticmethod
    # def find_oy(vector_co, vector_ox, square_of_base_length):
    #    vector_c = [vector_ox[2] * vector_co[1] - vector_ox[1] * vector_co[2],
    #                vector_ox[0] * vector_co[2] - vector_ox[2] * vector_co[0],
    #                vector_ox[0] * vector_co[1] - vector_ox[1] * vector_co[0]]
    #    return multiply_vector_on_scalar(vector_c, math.sqrt(square_of_base_length) / size_of_vector(vector_c))

    def find_ox(self, vector_co, vector_cf, vector_oy, stretch_ratio_of_cf, square_of_base_length):
        radius_vector_to_f_on_camera = sum_of_vectors(minus_vector(vector_co),
                                                      multiply_vector_on_scalar(vector_cf, stretch_ratio_of_cf))
        vector_oy_direction = sum_of_vectors(multiply_vector_on_scalar(radius_vector_to_f_on_camera,
                                                                       math.sqrt(square_of_base_length)),
                                             minus_vector(multiply_vector_on_scalar(vector_oy, self.f_y)))
        return multiply_vector_on_scalar(vector_oy_direction, 1 / self.f_x)

    def count_target(self, vector_co, vector_cf, vector_cs, stretch_ratio_of_cf,
                     stretch_ratio_of_cs, square_of_base_length):

        vector_oy = self.find_oy(vector_co, vector_cf, vector_cs, stretch_ratio_of_cf, stretch_ratio_of_cs,
                                 square_of_base_length)
        vector_ox = self.find_ox(vector_co, vector_cf, vector_oy, stretch_ratio_of_cf, square_of_base_length)
        # print("vector_ox")
        # print(vector_ox)
        # print("vector_oy")
        # print(vector_oy)
        # print("m")
        # print(math.sqrt(square_of_base_length))
        vector_oy_ = self.find_oy_through_rotate(vector_ox, vector_co, math.sqrt(square_of_base_length))

        print("oy")
        print(vector_oy)
        print("oy_")
        print(vector_oy_)
        print("ox")
        print(vector_ox)

        vector_to_target = sum_of_vectors(vector_co,
                                          sum_of_vectors(
                                              multiply_vector_on_scalar(vector_ox, self.target_x),
                                              multiply_vector_on_scalar(vector_oy, self.target_y)))
        return vector_to_target

    @staticmethod
    def find_ox_through_rotate(radius_vector_to_f_on_camera, base_length):
        return multiply_vector_on_scalar(radius_vector_to_f_on_camera,
                                         base_length/size_of_vector(radius_vector_to_f_on_camera))

    @staticmethod
    def find_oy_through_rotate(vector_ox, vector_co, base_length):
        vector_oy_direction = [vector_ox[1]*vector_co[2] - vector_ox[2]*vector_co[1],
                               vector_ox[2]*vector_co[0] - vector_ox[0]*vector_co[2],
                               vector_ox[0]*vector_co[1] - vector_ox[1]*vector_co[0]]
        return multiply_vector_on_scalar(vector_oy_direction, base_length/size_of_vector(vector_oy_direction))

    def find_target(self, vector_co, vector_cf, stretch_ratio_of_cf, square_of_base_length):
        rotate_angle = self.f_y / (math.sqrt(self.f_x ** 2 + self.f_y ** 2))
        new_f = sum_of_vectors(vector_cf, minus_vector(self.get_coordinates()))
        new_o = sum_of_vectors(vector_co, minus_vector(self.get_coordinates()))
        rotated_cf = rotate(new_f, [0.0, 0.0, 0.0], new_o, rotate_angle)
        vector_cf = sum_of_vectors(rotated_cf, self.get_coordinates())
        radius_vector_to_f_on_camera = sum_of_vectors(minus_vector(vector_co),
                                                      multiply_vector_on_scalar(vector_cf, stretch_ratio_of_cf))
        vector_ox = self.find_ox_through_rotate(radius_vector_to_f_on_camera, math.sqrt(square_of_base_length))
        vector_oy = self.find_oy_through_rotate(vector_ox, vector_co, math.sqrt(square_of_base_length))

        print("oy rotate")
        print(vector_oy)
        print("ox rotate")
        print(vector_ox)

        vector_to_target = sum_of_vectors(vector_co,
                                          sum_of_vectors(
                                              multiply_vector_on_scalar(vector_ox, self.target_x),
                                              multiply_vector_on_scalar(minus_vector(vector_oy), self.target_y)))
        return vector_to_target



# TODO: rewrite on numpy
def scalar_product(f_vector, s_vector):
    product = 0
    if len(f_vector) == len(s_vector):
        for i in range(len(f_vector)):
            product += f_vector[i] * s_vector[i]
    return product


def minus_vector(vector):
    product = []
    for val in vector:
        product.append(-val)
    return product


def sum_of_vectors(f_vector, s_vector):
    product = []
    if len(f_vector) == len(s_vector):
        for i in range(len(f_vector)):
            product.append(f_vector[i] + s_vector[i])
    return product


def multiply_vector_on_scalar(vector, scalar):
    product = []
    for i in range(len(vector)):
        product.append(vector[i] * scalar)
    return product


def size_of_vector(vector):
    return math.sqrt(scalar_product(vector, vector))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_vector(vector, rotate_angle) -> []:
    return [vector[0] * math.cos(-rotate_angle) - vector[1] * math.sin(-rotate_angle),
            vector[0] * math.sin(-rotate_angle) + vector[1] * math.cos(-rotate_angle)]


# find roots of ax^2+bx+c=0, where a not 0
def quadratic_equation_roots(a: float, b: float, c: float) -> []:
    discr = b ** 2 - 4 * a * c
    if discr > 0.000001:
        x1 = (-b + math.sqrt(discr)) / (2 * a)
        x2 = (-b - math.sqrt(discr)) / (2 * a)
        return [x1, x2]
    elif -0.00001 < discr < 0.000001:
        x = -b / (2 * a)
        return [x, None]
    else:
        return [None, None]


def linear_equation_root(b: float, c: float):
    if b == 0:
        return [None]
    return [-c / b]


def solve_quadratic_equation(a: float, b: float, c: float) -> []:
    if a == 0:
        return linear_equation_root(b, c)
    else:
        return quadratic_equation_roots(a, b, c)


def find_coordinates_of_target(first_camera: Camera, second_camera: Camera):
    vector_cameras = sum_of_vectors(first_camera.get_coordinates(), minus_vector(second_camera.get_coordinates()))
    vector_f_target = first_camera.vector_to_target
    vector_s_target = second_camera.vector_to_target

    def calculate(vector_f, vector_s, vector_c):
        # TODO: rename variables
        R = vector_f[0] ** 2 + vector_f[1] ** 2 + vector_f[2] ** 2
        G = vector_s[0] ** 2 + vector_s[1] ** 2 + vector_s[2] ** 2
        K = vector_c[0] * vector_s[0] + vector_c[1] * vector_s[1] + \
            vector_c[2] * vector_s[2]
        M = vector_f[0] * vector_s[0] + vector_f[1] * vector_s[1] + \
            vector_f[2] * vector_s[2]
        L = vector_c[0] * vector_f[0] + vector_c[1] * vector_f[1] + \
            vector_c[2] * vector_f[2]

        stretch_of_second_vector = (R * K - M * L) / (R * G - (M ** 2))
        stretch_of_first_vector = -(L - stretch_of_second_vector * M) / R
        return stretch_of_first_vector, stretch_of_second_vector

    stretch_of_first_vector, stretch_of_second_vector = calculate(vector_f_target, vector_s_target, vector_cameras)
    ans_f = multiply_vector_on_scalar(sum_of_vectors(
        sum_of_vectors(
            multiply_vector_on_scalar(vector_f_target, stretch_of_first_vector), first_camera.get_coordinates()
        ),
        sum_of_vectors(
            multiply_vector_on_scalar(vector_s_target, stretch_of_second_vector), second_camera.get_coordinates()
        )), 0.5)

    # TODO
    # TODO
    vector_f_target = first_camera.vector_to_targetD
    vector_s_target = second_camera.vector_to_targetD
    stretch_of_first_vector, stretch_of_second_vector = calculate(vector_f_target, vector_s_target, vector_cameras)
    ans_s = multiply_vector_on_scalar(sum_of_vectors(
        sum_of_vectors(
            multiply_vector_on_scalar(vector_f_target, stretch_of_first_vector), first_camera.get_coordinates()
        ),
        sum_of_vectors(
            multiply_vector_on_scalar(vector_s_target, stretch_of_second_vector), second_camera.get_coordinates()
        )), 0.5)

    vector_f_target = first_camera.vector_to_target
    vector_s_target = second_camera.vector_to_targetD
    stretch_of_first_vector, stretch_of_second_vector = calculate(vector_f_target, vector_s_target, vector_cameras)
    ans_t = multiply_vector_on_scalar(sum_of_vectors(
        sum_of_vectors(
            multiply_vector_on_scalar(vector_f_target, stretch_of_first_vector), first_camera.get_coordinates()
        ),
        sum_of_vectors(
            multiply_vector_on_scalar(vector_s_target, stretch_of_second_vector), second_camera.get_coordinates()
        )), 0.5)

    vector_f_target = first_camera.vector_to_targetD
    vector_s_target = second_camera.vector_to_target
    stretch_of_first_vector, stretch_of_second_vector = calculate(vector_f_target, vector_s_target, vector_cameras)
    ans_a = multiply_vector_on_scalar(sum_of_vectors(
        sum_of_vectors(
            multiply_vector_on_scalar(vector_f_target, stretch_of_first_vector), first_camera.get_coordinates()
        ),
        sum_of_vectors(
            multiply_vector_on_scalar(vector_s_target, stretch_of_second_vector), second_camera.get_coordinates()
        )), 0.5)
    # TODO: write ternary search
    return [ans_f, ans_s, ans_t, ans_a]


# TODO
# TODO
def rot(theta, u):
    return [[cos(theta) + u[0] ** 2 * (1 - cos(theta)),
             u[0] * u[1] * (1 - cos(theta)) - u[2] * sin(theta),
             u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
            [u[0] * u[1] * (1 - cos(theta)) + u[2] * sin(theta),
             cos(theta) + u[1] ** 2 * (1 - cos(theta)),
             u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
            [u[0] * u[2] * (1 - cos(theta)) - u[1] * sin(theta),
             u[1] * u[2] * (1 - cos(theta)) + u[0] * sin(theta),
             cos(theta) + u[2] ** 2 * (1 - cos(theta))]]


def rotate(pointToRotate, point1, point2, theta):
    u = []
    squaredSum = 0
    for i, f in zip(point1, point2):
        u.append(f - i)
        squaredSum += (f - i) ** 2

    u = [i / squaredSum for i in u]

    r = rot(-theta, u)
    rotated = []

    for i in range(3):
        rotated.append(round(sum([r[j][i] * pointToRotate[j] for j in range(3)])))

    return rotated


def find_vector_co(stretch_ratio_of_cf, vector_cf, stretch_ratio_of_cs, vector_cs):
    vector_co = [None, None, None]
    vector_coD = [None, None, None]

    matrix_to_find_co = \
        [[stretch_ratio_of_cf * vector_cf[0], stretch_ratio_of_cf * vector_cf[1],
          stretch_ratio_of_cf * vector_cf[2],
          1],
         [stretch_ratio_of_cs * vector_cs[0], stretch_ratio_of_cs * vector_cs[1],
          stretch_ratio_of_cs * vector_cs[2],
          1]]
    if abs(matrix_to_find_co[0][0]) < 0.00001:
        buf = matrix_to_find_co[0]
        matrix_to_find_co[0] = matrix_to_find_co[1]
        matrix_to_find_co[1] = buf
    if abs(matrix_to_find_co[0][0]) < 0.00001:
        if abs(matrix_to_find_co[0][1]) < 0.00001:
            buf = matrix_to_find_co[0]
            matrix_to_find_co[0] = matrix_to_find_co[1]
            matrix_to_find_co[1] = buf
        if abs(matrix_to_find_co[0][1]) < 0.00001:
            # both of point on one line
            assert 0
        else:
            buf = matrix_to_find_co[1][1]
            for i in range(4):
                matrix_to_find_co[1][i] -= matrix_to_find_co[0][i] * buf / matrix_to_find_co[0][1]
            if abs(matrix_to_find_co[1][2]) <= 0.00001:
                # both of point on one line
                assert 0
            else:
                z = matrix_to_find_co[1][3] / matrix_to_find_co[1][2]
                y = (matrix_to_find_co[0][3] - matrix_to_find_co[0][2] * z) / matrix_to_find_co[0][1]
                x = math.sqrt(1 - y ** 2 - z ** 2)
                vector_co = [x, y, z]
                vector_coD = [-x, y, z]
                many_solutions = 1
    else:
        buf = matrix_to_find_co[1][0]
        for i in range(4):
            matrix_to_find_co[1][i] -= matrix_to_find_co[0][i] * buf / matrix_to_find_co[0][0]

        if abs(matrix_to_find_co[1][1]) < 0.00001:
            if abs(matrix_to_find_co[1][2]) < 0.00001:
                # both of point on one line
                assert 0
            else:
                buf = matrix_to_find_co[0][2]
                for i in range(2):
                    matrix_to_find_co[0][i + 2] -= matrix_to_find_co[1][i + 2] * buf / matrix_to_find_co[1][2]
                y_roots = solve_quadratic_equation(matrix_to_find_co[0][1] ** 2 + matrix_to_find_co[0][0] ** 2,
                                                   -2.0 * (matrix_to_find_co[0][3] * matrix_to_find_co[0][1] *
                                                           matrix_to_find_co[0][0]),
                                                   (matrix_to_find_co[0][0] * matrix_to_find_co[1][3]) ** 2 +
                                                   matrix_to_find_co[0][3] ** 2 - matrix_to_find_co[0][0] ** 2
                                                   )
                y = y_roots[0]
                x = (matrix_to_find_co[0][3] - matrix_to_find_co[0][1] * y) / matrix_to_find_co[0][0]
                z = matrix_to_find_co[1][3] / matrix_to_find_co[1][2]
                vector_co = [x, y, z]
                if not (y_roots[1] is None):
                    many_solutions = 1
                    y = y_roots[1]
                    x = (matrix_to_find_co[0][3] - matrix_to_find_co[0][1] * y) / matrix_to_find_co[0][0]
                    z = matrix_to_find_co[1][3] / matrix_to_find_co[1][2]
                    vector_coD = [x, y, z]
        else:
            buf = matrix_to_find_co[0][1]
            for i in range(3):
                matrix_to_find_co[0][i + 1] -= matrix_to_find_co[1][i + 1] * buf / matrix_to_find_co[1][1]

            # print(matrix_to_find_co)
            # TODO: what does it mean: two solutions, how we should choose?
            z_roots = solve_quadratic_equation((matrix_to_find_co[0][2] * matrix_to_find_co[1][1]) ** 2 +
                                               (matrix_to_find_co[1][2] * matrix_to_find_co[0][0]) ** 2 +
                                               (matrix_to_find_co[0][0] * matrix_to_find_co[1][1]) ** 2,
                                               -2.0 * (matrix_to_find_co[0][3] * matrix_to_find_co[0][2] *
                                                       (matrix_to_find_co[1][1] ** 2) +
                                                       matrix_to_find_co[1][3] * matrix_to_find_co[1][2] *
                                                       (matrix_to_find_co[0][0] ** 2)),
                                               (matrix_to_find_co[0][3] * matrix_to_find_co[1][1]) ** 2 +
                                               (matrix_to_find_co[1][3] * matrix_to_find_co[0][0]) ** 2 -
                                               (matrix_to_find_co[1][1] * matrix_to_find_co[0][0]) ** 2)
            x = (matrix_to_find_co[0][3] - z_roots[0] * matrix_to_find_co[0][2]) / matrix_to_find_co[0][0]
            y = (matrix_to_find_co[1][3] - z_roots[0] * matrix_to_find_co[1][2]) / matrix_to_find_co[1][1]
            vector_co = [x, y, z_roots[0]]

            if not (z_roots[1] is None):
                many_solutions = 1
                x = (matrix_to_find_co[0][3] - z_roots[1] * matrix_to_find_co[0][2]) / matrix_to_find_co[0][0]
                y = (matrix_to_find_co[1][3] - z_roots[1] * matrix_to_find_co[1][2]) / matrix_to_find_co[1][1]
                vector_coD = [x, y, z_roots[1]]
    return vector_co, vector_coD


if __name__ == "__main__":
    a = [[0.5639186855807241, -0.022822905406525134, -0.08996509972152193],
         [0.5388528849583776, 0.050639974891856, -0.15798646304195235],
         [0.5386274807863911, 0.0384262532933352, -0.15198890476865973],
         [0.5649022181054215, -0.01396682022919598, -0.08919381969724212]]
    b = [[0.022251722171753058, -0.48782809910903957, 0.20797589757965157],
         [-0.009801640443955276, -0.6337750633607904, 0.2969417029533727],
         [0.04066177935308343, -0.593541836284741, 0.25581246549353004],
         [-0.017342057142611618, -0.5138781128778385, 0.2383222567939116]]


    def find_size(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


    #for i in a:
    #    for j in b:
    #        print(find_size(i, j))

    first_reference_point_coordinates_f = [-695.0, 25.0]
    second_reference_point_coordinates_f = [397.0, 166.0]
    target_coordinates_f = [-695.0, 25.0]
    first_reference_point_coordinates_s = [-712.0, 53.0]
    second_reference_point_coordinates_s = [709.0, 164.0]
    target_coordinates_s = [-712.0, 55.0]

    first_reference_point_coordinates = ['-0.47', '3.31', '-0.17']
    second_reference_point_coordinates = ['1.16', '5.56', '-0.06']
    first_camera_coordinates = ['0', '0', '0']
    second_camera_coordinates = ['1.16', '0', '0']
