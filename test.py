import geometry
import math


def test_rotate_vector():
    def almost_equal(value_1, value_2, accuracy=10 ** -8):
        return abs(value_1[0] - value_2[0]) + abs(value_1[1] - value_2[1]) < accuracy
    f_y = float(3)
    f_x = float(4)
    rotate_angle = math.asin(f_y / (math.sqrt(f_x ** 2 + f_y ** 2)))
    assert almost_equal(geometry.rotate_vector([f_x, f_y], rotate_angle),  [5.0, 0.0])

    f_y = float(4)
    f_x = float(3)
    rotate_angle = math.asin(f_y / (math.sqrt(f_x ** 2 + f_y ** 2)))
    assert almost_equal(geometry.rotate_vector([f_x, f_y], rotate_angle), [5.0, 0.0])

    f_y = float(2)
    f_x = float(2)
    rotate_angle = math.asin(f_y / (math.sqrt(f_x ** 2 + f_y ** 2)))
    assert almost_equal(geometry.rotate_vector([f_x, f_y], rotate_angle), [math.sqrt(8), 0.0])


def test_find_stretch():

    f_point = geometry.KnownPoint(4, 5, 5)
    s_point = geometry.KnownPoint(5, -3, 0)
    camera = geometry.Camera(2, 3, 3, -1, 1, 2, -1, 2, 2)

    assert camera.find_stretch_ratio_and_pixel_size(f_point, s_point) == (0.5, 1/3, 1.0)


def test_scalar_product():
    assert geometry.scalar_product([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert geometry.scalar_product([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 3.0

if __name__ == "__main__":
    test_rotate_vector()
    test_find_stretch()
    test_scalar_product()
    print("Done!")