import csv
import re
import math
import numpy as np
import os
import pandas as pd
from gmpy2 import mpfr, log2, get_context, gmpy2, exp2,fma
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

from 强化学习.表达式重写生成Tools import geneExprs


def uniform_distribution_inclusive(left_endpoint, right_endpoint, size):
    """
    在指定区间 [left_endpoint, right_endpoint] 内均匀生成 size 个点，包括右端点。

    :param left_endpoint: 区间的左端点
    :param right_endpoint: 区间的右端点
    :param size: 生成的点的数量
    :return: 包含均匀分布的点的列表
    """
    if size < 2:
        raise ValueError("size 必须大于 1，以确保包含右端点")
    step = (right_endpoint - left_endpoint) / (size)
    vec = [left_endpoint + i * step for i in range(size)]
    return vec

def herbie_error(origin, oracle) :
    """
    计算Herbie风格的浮点误差度量

    参数:
        x: 近似结果
        y: 精确结果

    返回:
        误差度量值 (以2为底的对数)
    """
    get_context().precision = 128
    if origin == oracle:
        return 0.0
    # 获取x和y之间的所有浮点数
    min_val = min(origin, oracle)
    max_val = max(origin, oracle)
    # 计算它们之间的浮点数数量
    count = 0
    current = mpfr(min_val)

    # 使用nextafter逐个遍历浮点数
    while current <= max_val:
        count += 1
        next_val = np.nextafter(current, math.inf)
        if next_val == current:  # 处理最大值情况
            break
        current = next_val
    return gmpy2.log2(current)

# 计算两个数的ullp差值，即两个数的ulp误差
def compute_ulp_diff(origin, oracle):
    get_context().precision = 128
    diff = mpfr(str(origin)) - mpfr(str(oracle))
    ulp_oracle = float(oracle)
    c = compute_ulp(ulp_oracle)
    ulp_diff = abs(diff / mpfr(c))
    return float(ulp_diff)

def compute_re(origin, oracle):
    get_context().precision = 128
    diff = mpfr(origin) - mpfr(oracle)
    re = diff / mpfr(oracle)
    print(f"相对误差:{re:.20e}")
# 计算单个参数的ulp值
def compute_ulp(y):
    x = abs(y)

    # Handle special cases
    if x == 0:
        return math.ldexp(1, -1074)  # Smallest denormal

    if x < math.ldexp(1, -1021):
        return math.ldexp(1, -1074)

    # Calculate upper bound safely
    max_normal_part1 = (1.0 - 2.0 ** -53)
    max_normal_part2 = math.ldexp(1.0, 1023)
    max_normal = max_normal_part1 * max_normal_part2 * 2.0  # (1-2^-53)*2^1024

    if x > max_normal or math.isinf(x):
        return math.ldexp(1, 971)  # ULP for infinity

    # Initialize binary search variables
    powermin = math.ldexp(1, -1021)
    expmin = -1021
    powermax = math.ldexp(1, 1023) * 2.0  # 2^1024
    expmax = 1024

    # Binary search
    while expmax - expmin > 1:
        # Calculate middle exponent
        if (expmin + expmax) % 2 == 0:
            expmiddle = (expmin + expmax) // 2
        else:
            expmiddle = (expmin + expmax + 1) // 2

        powermiddle = math.ldexp(1, expmiddle)

        if x >= powermiddle:
            powermin = powermiddle
            expmin = expmiddle
        else:
            powermax = powermiddle
            expmax = expmiddle

    # Determine final ULP value
    if x == powermin:
        return math.ldexp(1, expmin - 53)
    else:
        return math.ldexp(1, expmin - 52)

# 生成均匀采样点
def uniform_sampling(left,right,nums):
    return np.linspace(left,right,nums)

def evaluate_expression(expr, x_val):
    """
    计算表达式在64位和128位精度下的值
    :param expr: 字符串类型的表达式
    :param x_val: 输入值（建议以字符串形式传入以保证精度）
    :return: (64位结果, 128位结果)
    """
    # 定义允许的高精度数学函数
    allowed_functions = {
        'sqrt': gmpy2.sqrt,
        'exp': gmpy2.exp,
        'log': gmpy2.log,
        'sin': gmpy2.sin,
        'cos': gmpy2.cos,
        'tan': gmpy2.tan,
        'log1p': gmpy2.log1p,
        'expm1': gmpy2.expm1,
        'pi': gmpy2.const_pi,
        'fma': gmpy2.fma,
        'e': lambda: gmpy2.exp(1),
        'E':lambda: gmpy2.exp(1)
    }

    # 安全替换数学函数（使用正则表达式精确匹配）
    for func in allowed_functions:
        pattern = r'\b' + re.escape(func) + r'(\s*)\('
        replacement = f"allowed_functions['{func}']\\1("
        expr = re.sub(pattern, replacement, str(expr))

    def calculate(precision):
        """根据指定精度计算表达式"""
        ctx = gmpy2.get_context()
        ctx.precision = precision

        try:
            # 创建对应精度的x值（建议输入x_val为字符串）
            x = mpfr(x_val)
            namespace = {
                'x': x,
                'allowed_functions': allowed_functions,
                'gmpy2': gmpy2

            }
            # 添加常量（每次重新生成以保证精度）
            namespace.update({
                'pi': allowed_functions['pi'](),
                'e': allowed_functions['e'](),
                'E': allowed_functions['E']()
            })
            return eval(expr, {}, namespace)
        except Exception as e:
            print(f"{precision}位计算错误: {e}")
            return None

    # 分别计算两种精度
    return (calculate(53), calculate(113))

def oneParaErrorDetect(expr, x ):
    f_float, f_mpfr = evaluate_expression(expr, x)
    ulp_diff = compute_ulp_diff(f_float,f_mpfr)
    return ulp_diff

def handle_error_point(vec,G=500,ULP=1.0):
    """
    处理误差点，将误差点添加到 vec_after 中，并返回 vec_after。
    :param vec: 输入的二维 NumPy 数组，包含采样点和对应的误差值
    :return: 处理后的 vec_after
    """

    vec_after = []
    # 遍历输入数组 vec，步长为 G
    for i in range(0, vec.shape[0], G):
        max_ulp = 0
        max_row = np.array([0.0, 0.0])
        # 在当前步长范围内查找最大 ULP 值
        for j in range(i, min(i + G, vec.shape[0])):
            if vec[j, 1] > max_ulp:
                max_ulp = vec[j, 1]
                max_row = vec[j]
        # 如果最大 ULP 值大于等于 ULP 阈值，则将该误差点添加到 vec_after 中
        if max_ulp >= ULP:
            # print(max_ulp)
            vec_after.append(max_row)

    vec_after = np.array(vec_after)
    # 找到 vec_after 中的最大误差值
    if vec_after.size > 0:
        max_error = np.max(vec_after[:, 1])
    else:
        max_error = 0
    # 输出最大误差值
    print(f"不做任何处理时输出的最大误差值: {max_error}")
    return vec_after

def gene_dbscan_file(vec_after):
    """
    先判断文件是否存在，若存在则删除重新创建并写入表头，若不存在则创建文件并写入表头，
    最后以追加模式写入数据。
    :param vec_after: 包含误差点的 NumPy 数组
    :return: None
    """
    file_name = "galaxy.csv"
    # 判断文件是否存在
    if os.path.exists(file_name):
        # 若文件存在，删除文件
        os.remove(file_name)
        print(f"文件 {file_name} 已存在，已删除该文件。")
    else:
        print(f"文件 {file_name} 不存在，将创建新文件。")

    # 打开文件，追加模式
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["x", "ulp"])

        # 写入数据行
        for row in vec_after:
            writer.writerow(row)

    print(f"数据已成功追加到文件 {file_name}")

def sliding_window_analysis(vec, window_size=400, step=100, ULP=1.0):
    """
    使用滑动窗口法分析误差点，返回处理后的误差点数组。
    :param vec: 输入的二维 NumPy 数组，包含采样点和对应的误差值
    :param window_size: 滑动窗口的大小，默认为 200
    :param step: 滑动窗口每次移动的步长，默认为 50
    :param ULP: ULP 阈值，默认为 1.0
    :return: 处理后的误差点数组
    """
    max_error_points = []
    num = vec.shape[0]
    for i in range(0, num - window_size + 1, step):
        window_results = vec[i:i + window_size]
        window_errors = window_results[:, 1]
        max_index = np.argmax(window_errors)
        max_ulp = window_errors[max_index]
        if max_ulp >= ULP:
            max_error_points.append(window_results[max_index])

    max_error_points = np.array(max_error_points)
    if max_error_points.size > 0:
        max_error = np.max(max_error_points[:, 1])
    else:
        max_error = 0
    print(f"滑动窗口分析后输出的最大误差值: {max_error}")
    return max_error_points

def find_appropriate_window(expr, left, right, target_points=1000, num=500000):
    window_size = 200
    step = 50
    samples = uniform_distribution_inclusive(left, right, num)
    results = np.zeros((num, 2))
    for i, x in enumerate(samples):
        results[i, 0] = x
        results[i, 1] = oneParaErrorDetect(expr, x)
    while True:
        max_error_points = sliding_window_analysis(results, expr, window_size, step)
        num_points = len(max_error_points)
        if num_points < target_points:
            step = max(1, step // 2)
        elif num_points > target_points:
            window_size = int(window_size * 1.1)
        else:
            break
    return window_size, step

def predict_interval(x1, q1, x2, q2, left_endpoint, right_endpoint):
    """
    预测区间。

    :param x1: 下界函数的初始值
    :param q1: 下界函数的比率
    :param x2: 上界函数的初始值
    :param q2: 上界函数的比率
    :param left_endpoint: 区间的左端点
    :param right_endpoint: 区间的右端点
    :return: 包含预测区间的列表
    """
    result = []
    q1_prev = q1 ** -31
    q2_prev = q2 ** -31
    threshold = 1e-06

    for i in range(-30, 100):
        q1_prev *= q1
        q2_prev *= q2
        lower = x1 * q1_prev
        upper = x2 * q2_prev

        if lower > upper:
            lower, upper = upper, lower  # 交换 lower 和 upper

        if upper < left_endpoint or lower > right_endpoint:
            continue

        lower = max(lower, left_endpoint)
        upper = min(upper, right_endpoint)

        if upper - lower >= threshold:
            result.append([lower, upper])

    return result

def detect_all_interval_get_maximum_error(vec_interval,expr):
    """
    检测所有区间中的最大误差点。

    :param vec_interval: 区间列表，每个区间是一个包含两个元素的列表 [left, right]
    :return: 包含最大误差点的 x 值和 ULP 值的列表
    """
    result = []  # 用于存储最终结果（最大误差点的 x 值和 ULP 值）
    temp = []  # 用于临时存储每个区间的最大误差点
    num = 500000
    # 遍历所有给定的区间
    for current_interval in vec_interval:
        vec_uniform = np.zeros((num, 2))  # 用于存储当前区间内的误差点
        left = current_interval[0]  # 当前区间的左边界
        right = current_interval[1]  # 当前区间的右边界
        inputx = uniform_distribution_inclusive(left, right, num)  # 在当前区间内生成 500000 个均匀分布的 x 值
        print(f"当前区间范围：（{left,right}）")
        for i, x in enumerate(inputx):
            vec_uniform[i, 0] = x  # 第一列存储采样点x
            vec_uniform[i, 1] = oneParaErrorDetect(expr, x)  # 第二列存储计算结果


        # 在当前区间的所有误差点中，找到 ULP 值最大的误差点
        max_ulp_row = max(vec_uniform, key=lambda row: row[1])  # 使用 ULP 值作为比较标准
        print(f"当前区间最大误差值：{max_ulp_row}")
        temp.append([max_ulp_row[0], max_ulp_row[1]])  # 将最大误差点的 x 值和 ULP 值存储到 temp 中

    # 在所有区间的最大误差点中，找到 ULP 值最大的误差点
    max_ulp = max(temp, key=lambda row: row[1])  # 使用 ULP 值作为比较标准
    result.append(max_ulp[0])  # x 值
    result.append(max_ulp[1])  # ULP 值

    return result

def evaluate_expression_original(expr,size,light,right,file_path="ulp_0.txt"):
    get_context().precision = 128
    samples = uniform_distribution_inclusive(light, right, size)
    # 计算
    print(f"原始表达式：{expr}")
    results = np.zeros((size, 2))  # 每行有2列，共num行
    for i, x in enumerate(samples):
        results[i, 0] = x  # 第一列存储采样点x
        results[i, 1] = oneParaErrorDetect(expr, x)  # 第二列存储计算结果
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 文件存在，先删除文件
        os.remove(file_path)
        print(f"文件 {file_path} 已存在，已删除。")
    else:
        # 文件不存在，不需要删除
        print(f"文件 {file_path} 不存在，将创建新文件。")
    # 打开文件，追加模式
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["x", "ulp"])
        # 写入数据行
        for row in results:
            writer.writerow(row)
    print(f"数据已成功追加到文件 {file_path}")

# 计算{x,ulp}的结果
def cache_evaluate(expr,oracle,size,light,right) :

    samples = uniform_distribution_inclusive(light, right, size)
    # 计算
    results = np.zeros((size, 2))  # 每行有2列，共num行
    for i, x in enumerate(samples):
        results[i, 0] = x  # 第一列存储采样点x
        results[i, 1] = compute_ulp_diff(str(compute_expr_res(expr,x,53)), oracle[i])  # 第二列存储计算结果
    return results

def evaluate_expression_optimized(expr,size,light=0.1,right=10000):
    get_context().precision = 128
    samples = uniform_distribution_inclusive(light, right, size)
    # 计算
    results = np.zeros((size, 2))  # 每行有2列，共num行
    for i, x in enumerate(samples):
        results[i, 0] = x  # 第一列存储采样点x
        results[i, 1] = oneParaErrorDetect(expr, x)  # 第二列存储计算结果
        # 文件路径
    file_path = 'ulp_0.txt'
    origin = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    # for i in range(len(origin)):
    #     ori = origin[i, 1]
    #     print(ori)
    # print( type(origin_res))
    avg_ulp_origin = np.mean(origin[:, 1])
    print(len(origin[:, 1]))
    # 计算优化后的平均 ULP 误差
    avg_ulp_res = np.mean(results[:, 1])
    print(len(results[:, 1]))
    # 计算平均 ULP 误差的提升
    ulp_improvement = avg_ulp_origin - avg_ulp_res

    # 计算平均 ULP 误差的百分比提升
    ulp_improvement_percentage = (ulp_improvement / avg_ulp_origin) * 100
    print(f"优化表达式: {expr} 优化效果为：{ulp_improvement_percentage:.2f}")
    return ulp_improvement_percentage

def compute_optimization_effect(ori_expr,opt_expr,size=100000,light=1/100,right=1/2):
    get_context().precision = 128
    return evaluate_expression_optimized(opt_expr, size, light, right)

def run(expr,expr_opt,size=100000,light=1/100,right=1/2):
    get_context().precision = 128
    evaluate_expression_original(expr, size, light, right)
    res = evaluate_expression_optimized(expr_opt, size, light, right)
    return res*100


def compute_expr_res(expr, x_val, precision) :
    """
    计算表达式128位精度下的值 作为oracle值。
    :param expr: 字符串类型的表达式
    :param x_val: 输入值（建议以字符串形式传入以保证精度）
    :return: (64位结果, 128位结果)
    """
    # 定义允许的高精度数学函数
    allowed_functions = {
        'sqrt': gmpy2.sqrt,
        'exp': gmpy2.exp,
        'log': gmpy2.log,
        'sin': gmpy2.sin,
        'cos': gmpy2.cos,
        'tan': gmpy2.tan,
        'log1p': gmpy2.log1p,
        'expm1': gmpy2.expm1,
        'pi': gmpy2.const_pi,
        'fma': gmpy2.fma,
        'e': lambda: gmpy2.exp(1),
        'E':lambda: gmpy2.exp(1)
    }

    # 安全替换数学函数（使用正则表达式精确匹配）
    for func in allowed_functions:
        pattern = r'\b' + re.escape(func) + r'(\s*)\('
        replacement = f"allowed_functions['{func}']\\1("
        expr = re.sub(pattern, replacement, str(expr))

    def calculate(precision):
        """根据指定精度计算表达式"""
        ctx = gmpy2.get_context()
        ctx.precision = precision

        try:
            # 创建对应精度的x值（建议输入x_val为字符串）
            x = mpfr(x_val)
            namespace = {
                'x': x,
                'allowed_functions': allowed_functions,
                'gmpy2': gmpy2

            }
            # 添加常量（每次重新生成以保证精度）
            namespace.update({
                'pi': allowed_functions['pi'](),
                'e': allowed_functions['e'](),
                'E': allowed_functions['E']()
            })
            return eval(expr, {}, namespace)
        except Exception as e:
            print(f"{precision}位计算错误: {e}")
            return None

    # 分别计算两种精度
    return calculate(precision)

# 保存原表达式的低精度计算值和oracle计算值。
def get_expr_res(expr,size,light,right,precision):
    samples = uniform_distribution_inclusive(light, right, size)
    results = []
    # 第一行x值，第二行y值。
    for i, x in enumerate(samples):
        results.append(str(compute_expr_res(expr, x, precision)))
    return results


def get_optimization_effect(new_expr,size,light,right):
    # 计算当前函数与oracle 的误差值。
    oracle = res_cache["oracle"]
    if new_expr in res_cache :
        new_expr_res = res_cache[new_expr]

    else :
        new_expr_res = get_expr_res(new_expr,size,light,right)
    ulp_diff = np.empty(len(oracle))
    for i in range(oracle):
        ulp_diff[i] = compute_ulp_diff(new_expr_res[i], oracle[i])

    results = np.zeros((size, 2))  # 每行有2列，共num行
    file_path = 'ulp_0.txt'
    origin = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    # for i in range(len(origin)):
    #     ori = origin[i, 1]
    #     print(ori)
    # print( type(origin_res))
    avg_ulp_origin = np.mean(origin[:, 1])
    print(len(origin[:, 1]))
    # 计算优化后的平均 ULP 误差
    avg_ulp_res = np.mean(results[:, 1])
    print(len(results[:, 1]))
    # 计算平均 ULP 误差的提升
    ulp_improvement = avg_ulp_origin - avg_ulp_res

    # 计算平均 ULP 误差的百分比提升
    ulp_improvement_percentage = (ulp_improvement / avg_ulp_origin) * 100
    print(f"优化表达式: {expr} 优化效果为：{ulp_improvement_percentage:.2f}")
    return ulp_improvement_percentage



expr = "(exp(x) - 1) / log(exp(x))"
# size = 50000
# light = 0.01
# right = 0.5
# oracle = None
res_cache = {}
# if __name__ == "__main__":
#     get_context().precision = 128
#     # exprs = geneExprs(expr)
#     exprs = ["(exp(x) - 1) / log(exp(x))","(exp(x) - 1)/log1p(exp(x) - 1)"]
#     nums = len(exprs)
#     index = 0
#     oracle = get_expr_res(expr,size,light,right,113)
#     evaluate_expression_original(expr,size,light,right)
#
#     for ex in exprs:
#         ulp_res = cache_evaluate(ex,oracle,size,light,right)
#         file_path = 'ulp_0.txt'
#         origin = np.genfromtxt(file_path, delimiter=',', skip_header=1)
#         evaluate_expression_original(ex,size,light,right,"ulp_1.txt")
#
#         avg_ulp_origin = np.mean(origin[:, 1])
#         avg_ulp_res = np.mean(ulp_res[:, 1])
#         ulp_improvement = avg_ulp_origin - avg_ulp_res
#         # 计算平均 ULP 误差的百分比提升
#         ulp_improvement_percentage = (ulp_improvement / avg_ulp_origin) * 100
#         print(f"优化表达式: {ex} 优化效果为：{ulp_improvement_percentage:.2f}")
#         index += 1