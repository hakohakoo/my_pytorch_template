import math
import numpy as np


def get_one_line(k=0, n=1024, x=0, y=0, z=100, d_length=1):
    dx = d_length * math.cos(k)
    dy = d_length * math.sin(k)
    start_x = 0 - n * dx / 2 + x
    end_x = n * dx / 2 + x
    start_y = 0 - n * dy / 2 + y
    end_y = n * dy / 2 + y
    rx = np.linspace(start_x, end_x, n)
    ry = np.linspace(start_y, end_y, n)
    rz = np.linspace(z, z, n)
    return rx, ry, rz


def get_circle_line_by_num(k=0, n=1024, x=0, y=0, z=0, d_length=1):
    """
    通过测线数据的数量
    生成圆周测线
    n为采样数量
    x，y，z为坐标
    dLength为采样间隔
    """
    r = n * d_length / 2 / math.pi
    result_x = np.ones(n)
    result_y = np.ones(n)
    result_z = np.ones(n)
    for i in range(n):
        result_x[i] = r * math.cos(i * 2 * math.pi / n + k) + x
        result_y[i] = r * math.sin(i * 2 * math.pi / n + k) + y
        result_z[i] = z
    return result_x, result_y, result_z


def get_circle_line_by_r(r=300, x=0, y=0, z=0, d_length=1):
    """
    通过半径
    生成圆周测线
    n为采样数量
    x，y，z为坐标
    dLength为采样间隔
    """
    n = int(2 * r * math.pi / d_length)
    result_x = np.ones(n)
    result_y = np.ones(n)
    result_z = np.ones(n)
    for i in range(n):
        result_x[i] = r * math.cos(i * 2 * math.pi / n) + x
        result_y[i] = r * math.sin(i * 2 * math.pi / n) + y
        result_z[i] = z
    return result_x, result_y, result_z
