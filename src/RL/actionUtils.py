import re
import itertools
import sympy as sp
from sympy import (sympify, expand, factor, cancel, simplify,
                   expand_trig, trigsimp, expand_log, logcombine,
                   expand_power_base, powsimp, powdenest, together,
                    fraction, symbols, apart, log, preorder_traversal, Function, exp
                   )
from sympy.codegen.cfunctions import log1p, expm1
from sympy.polys.polytools import Poly

def smart_rewrite(expr):
    """智能重写表达式为更稳定的形式"""
    # 先尝试标准简化
    expr = sp.together(expr)
    x = sp.symbols('x')

    # 处理分式情况
    if expr.is_rational_function(x):
        numer, denom = sp.fraction(expr)

        # 确保分子和分母都是多项式
        numer_poly = Poly(numer, x)
        denom_poly = Poly(denom, x)

        # 情况1: x/(x+c) -> 1 - c/(x+c)
        if numer_poly == Poly(x, x) and denom_poly.has(x):
            c = denom_poly - Poly(x, x)
            if c != 0:
                return 1 - c / denom_poly

        # 情况2: (ax+b)/(cx+d) -> a/c + (b - a*d/c)/(cx+d)
        if denom_poly.degree() == 1 and numer_poly.degree() == 1:
            a = numer_poly.coeff_monomial(x ** 1)
            b = numer_poly.coeff_monomial(x ** 0)
            c = denom_poly.coeff_monomial(x ** 1)
            d = denom_poly.coeff_monomial(x ** 0)
            if c != 0:
                return a / c + (b - a * d / c) / denom_poly

    return expr
# log1p
def optimize_log1p(expr):
    expr = sympify(expr)
    # 如果表达式是 log(expr)，则替换为 log1p(expr - 1)
    if expr.func == log:
        return log1p(optimize_log1p(expr.args[0]) - 1)
    # 如果表达式是复合表达式，递归优化其子表达式
    elif expr.args:
        return expr.func(*[optimize_log1p(arg) for arg in expr.args])
    # 如果表达式是原子（如符号或数字），直接返回
    else:
        return expr

# expm1
def optimize_expm1(expr):
    # 如果表达式是 exp(expr) - 1，则替换为 expm1(expr)
    if expr.is_Add and len(expr.args) == 2 and expr.args[1] == -1 and expr.args[0].func == exp:
        return expm1(optimize_expm1(expr.args[0].args[0]))
    # 如果表达式是 exp(expr)，则替换为 expm1(expr) + 1
    elif expr.func == exp:
        return expm1(optimize_expm1(expr.args[0])) + 1
    # 如果表达式是复合表达式，递归优化其子表达式
    elif expr.args:
        return expr.func(*[optimize_expm1(arg) for arg in expr.args])
    # 如果表达式是原子（如符号或数字），直接返回
    else:
        return expr

#log(exp)
def log_exp_replace(expr):
    # 如果表达式是 log(exp(expr))，则替换为 expr
    if expr.func == sp.log and expr.args[0].func == sp.exp:
        return expr.args[0].args[0]
    # 如果表达式是复合表达式，递归优化其子表达式
    elif expr.args:
        return expr.func(*[log_exp_replace(arg) for arg in expr.args])
    # 如果表达式是原子（如符号或数字），直接返回
    else:
        return expr
# 结合律
def associative_law(expr):
    """
    生成表达式的多种结合方式。
    :param expr: 输入的表达式
    :return: 一个列表，包含所有可能的结合方式
    """
    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        if len(args) <= 2:
            return [expr]
        combinations_list = []
        for i in range(1, len(args)):
            left = sp.Mul(*args[:i], evaluate=False)
            right = sp.Mul(*args[i:], evaluate=False)
            combined_expr = sp.Mul(left, right, evaluate=False)
            combinations_list.append(combined_expr)
            # 递归生成左半部分和右半部分的组合
            left_combinations = associative_law(left)
            right_combinations = associative_law(right)
            for lc in left_combinations:
                for rc in right_combinations:
                    combinations_list.append(sp.Mul(lc, rc, evaluate=False))
        return combinations_list
    return [expr]

# 分配律
def custom_distribute(expr):
    # 如果表达式是乘法，并且其中一个因子是加法
    if isinstance(expr, sp.Mul):
        factors = list(expr.args)
        for i, factor in enumerate(factors):
            if isinstance(factor, sp.Add):
                # 分配律展开
                distributed_terms = []
                for term in factor.args:
                    new_expr = sp.Mul(*([term] + factors[:i] + factors[i + 1:]))
                    distributed_terms.append(custom_distribute(new_expr))
                return sp.Add(*distributed_terms)

    # 如果表达式是加法，递归处理每个加数
    elif isinstance(expr, sp.Add):
        return sp.Add(*[custom_distribute(arg) for arg in expr.args])

    # 如果表达式是除法，转换为乘法
    elif isinstance(expr, sp.Pow) and expr.exp == -1:
        base = expr.base
        return sp.Mul(1, sp.Pow(base, -1))

    # 如果表达式是其他类型，直接返回
    return expr

# 简单处理sqrt（x+a）±sqrt（x+b）的情况
def sqrt_rewrite(expression):
    expression = str(expression)
    # 正则表达式匹配 sqrt 和其内部表达式
    sqrt_pattern = r"sqrt\(([^)]+)\)"
    matches = re.findall(sqrt_pattern, expression)
    # 提取两个 sqrt 内部的表达式
    sqrt_expr1, sqrt_expr2 = matches
    x = symbols('x')
    sqrt1 = sympify(sqrt_expr1)
    sqrt2 = sympify(sqrt_expr2)
    expr_res = sqrt1 - sqrt2
    if not expr_res.is_constant() :
        return expression

    sqrt_expr_sub1 = sympify(f"sqrt({sqrt_expr1}) - sqrt({sqrt_expr2})")
    sqrt_expr_sub2 = sympify(f"sqrt({sqrt_expr2}) - sqrt({sqrt_expr1})")
    sqrt_expr_add1 = sympify(f"sqrt({sqrt_expr1}) + sqrt({sqrt_expr2})")

    sqrt_expr = sympify(expression)
    if simplify(sqrt_expr_sub1 - sqrt_expr) == 0 or simplify(sqrt_expr_sub2 - sqrt_expr) == 0:
        # 构造共轭表达式并化简
        sqrt_expr_new = sympify(expr_res/sqrt_expr_add1)
        return sqrt_expr_new
    if simplify(sqrt_expr_add1 - sqrt_expr) == 0:
        # 构造共轭表达式并化简
        sqrt_expr_new = sympify(expr_res/sqrt_expr_sub1)
        return sqrt_expr_new
    return expression
# 提取表达式中包含的函数
def extract_functions(expr_str):
    math_functions = {
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'exp', 'log', 'sqrt', 'cbrt', 'pow'
    }
    # 使用正则表达式匹配函数名
    pattern = r'\b(' + '|'.join(math_functions) + r')\b'

    found_functions = set(re.findall(pattern, expr_str))
    return sorted(found_functions)

# 分类表达式函数类型
def classification_function(expr, found_functions):
    # 默认基础多项式的合并与展开
    simplification_functions = set()
    simplification_functions.update(
        [associative_law, custom_distribute, smart_rewrite, expand, factor, cancel])

    if not found_functions:
        simplification_functions.add(sp.apart)
        return list(simplification_functions)

    for func in found_functions:
        if func in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
            simplification_functions.update([expand_trig, trigsimp])
        elif func in ['log']:
            simplification_functions.update([expand_log, logcombine, optimize_log1p, log_exp_replace])
        elif func in ['exp']:
            simplification_functions.update([optimize_expm1, log_exp_replace])
        elif func in ['sqrt']:
            sqrt_pattern = r"sqrt\(([^)]+)\)"
            matches = re.findall(sqrt_pattern, str(expr))
            # 检查 sqrt 的数量是否为 2
            if len(matches) == 2:
                simplification_functions.add(sqrt_rewrite)
        elif func in ['pow']:
            simplification_functions.update([expand_power_base, powsimp, powdenest])

    return list(simplification_functions)
# 应用优化函数
def apply_simplifications(expr, simplification_functions):
    expr = sympify(expr)
    results = set()
    results.add(str(expr))  # 添加原始表达式
    for func in simplification_functions:
        try:
            result = func(expr)
            results.add(str(result))
        except Exception as e:
            results.add(f"Error: {str(e)}")
    return results


# 生成所有可能的函数排列组合
# def generate_combinations(simplification_functions):
#     combinations = []
#     nums = len(simplification_functions)
#     print(nums)
#     for i in range(1, nums + 1):
#         # 使用 itertools.permutations 生成排列
#         permutations_list = itertools.permutations(simplification_functions, i)
#         combinations.extend(permutations_list)
#     print(len(combinations))
#     return combinations


# 生成所有可能的函数组合
def generate_combinations(simplification_functions):
    combinations = []
    nums = len(simplification_functions)
    for i in range(nums):
        combinations.extend(itertools.combinations(simplification_functions, i+1))
    print(len(combinations))
    return combinations

# 应用所有组合
def apply_all_combinations(expr, simplification_functions):
    results = {}  # 使用字典存储结果，键为表达式，值为变换策略列表
    # 生成所有可能的函数组合
    combinations = generate_combinations(simplification_functions)
    for combination in combinations:
        # 初始化表达式列表，将输入表达式转换为列表形式
        temp_expr_list = [(sympify(expr), [])]  # 每个元素是一个元组，包含表达式和变换策略列表
        for func in combination:
            new_expr_list = []
            for temp_expr, strategy in temp_expr_list:
                try:
                    if func == expand_log:
                        result = func(temp_expr, force=True)
                    elif func == logcombine:
                        result = func(temp_expr, force=True)
                    elif func == expand_power_base:
                        result = func(temp_expr, force=True)
                    else:
                        result = func(temp_expr)
                    new_strategy = strategy + [func.__name__]  # 更新变换策略
                    # 如果结果是列表，将列表中的每个元素添加到新的表达式列表中
                    if isinstance(result, list):
                        for item in result:
                            new_expr_list.append((item, new_strategy))
                    else:
                        new_expr_list.append((result, new_strategy))
                except Exception as e:
                    # 处理异常情况，将错误信息作为结果添加到新的表达式列表中
                    new_expr_list.append((f"Error: {str(e)}", strategy))
            # 更新当前表达式列表
            temp_expr_list = new_expr_list
        # 将最终处理后的表达式列表中的每个元素添加到结果集中
        for final_expr, final_strategy in temp_expr_list:
            expr_str = str(final_expr)
            if expr_str not in results:
                results[expr_str] = final_strategy
    return results

def geneExprs(expr):
    found_functions = extract_functions(expr)
    simplification_functions = classification_function(expr,found_functions)
    exprs = apply_all_combinations(expr, simplification_functions)
    for result, strategy in exprs.items():
        print(f"表达式: {result}，变换策略: {strategy}")
    return exprs

# 主函数
def main():
    expr = "log(1 - x)/log(x + 1)" # x>0"
    found_functions = extract_functions(expr)
    simplification_functions = classification_function(expr, found_functions)

    results = apply_all_combinations(expr, simplification_functions)
    for result, strategy in results.items():
        print(f"表达式: {result}，变换策略: {strategy}")

def signal_op():
    expr = "log(1 - x)/log(x + 1)"  # x>0"
    print(optimize_log1p(expr))

if __name__ == "__main__":
    # main()
    signal_op()
