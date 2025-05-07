from sympy import (
    sympify, Basic, Symbol, Integer, Float, Rational,
    Add, Mul, Pow, Function, cos, sin, exp, log,  # 可根据需要扩展
    S  # SymPy 的预定义常量（如 S.One, S.NegativeOne）
)


def expression_to_ast(expr):
    """
    将 SymPy 表达式转换为可还原的 AST 结构。
    返回格式：(node_type, children_or_value)
    """
    expr = sympify(expr)  # 确保输入是 SymPy 表达式

    def _to_ast(node):
        if isinstance(node, Basic):
            if node.is_Symbol:
                return ('Symbol', str(node))  # 存储符号名称
            elif node.is_Integer:
                return ('Integer', int(node))  # 存储整数值
            elif node.is_Float:
                return ('Float', float(node))  # 存储浮点数值
            elif node.is_Rational:
                return ('Rational', (int(node.p), int(node.q)))  # 存储分数 p/q
            elif node is S.One:
                return ('One', None)  # SymPy 的 1
            elif node is S.NegativeOne:
                return ('NegativeOne', None)  # SymPy 的 -1
            elif node.is_Add or node.is_Mul or node.is_Pow:
                # 处理 Add, Mul, Pow 等运算符
                return (type(node).__name__, [_to_ast(arg) for arg in node.args])
            elif isinstance(node, Function):
                # 处理函数，如 cos, sin, exp, log 等
                return (type(node).__name__, [_to_ast(arg) for arg in node.args])
            else:
                raise ValueError(f"Unsupported SymPy node: {node}")
        else:
            # 如果不是 SymPy 类型（如 Python 的 int, float, str），直接存储
            return ('Literal', node)

    return _to_ast(expr)


def ast_to_expression(ast):
    """
    将 AST 结构还原为 SymPy 表达式。
    """
    node_type, payload = ast

    if node_type == 'Symbol':
        return Symbol(payload)  # 从字符串恢复符号
    elif node_type == 'Integer':
        return Integer(payload)  # 恢复整数
    elif node_type == 'Float':
        return Float(payload)  # 恢复浮点数
    elif node_type == 'Rational':
        p, q = payload
        return Rational(p, q)  # 恢复分数
    elif node_type == 'One':
        return S.One  # SymPy 的 1
    elif node_type == 'NegativeOne':
        return S.NegativeOne  # SymPy 的 -1
    elif node_type == 'Literal':
        return payload  # 直接返回 Python 的 int/float/str
    elif node_type in ('Add', 'Mul', 'Pow'):
        # 递归处理子节点
        args = [ast_to_expression(child) for child in payload]
        if node_type == 'Add':
            return Add(*args)
        elif node_type == 'Mul':
            return Mul(*args)
        elif node_type == 'Pow':
            return Pow(*args)
    elif node_type in ('cos', 'sin', 'exp', 'log'):  # 可扩展更多函数
        args = [ast_to_expression(child) for child in payload]
        if node_type == 'cos':
            return cos(*args)
        elif node_type == 'sin':
            return sin(*args)
        elif node_type == 'exp':
            return exp(*args)
        elif node_type == 'log':
            return log(*args)
    else:
        raise ValueError(f"Unknown AST node type: {node_type}")


# 测试代码
if __name__ == "__main__":
    from sympy import simplify

    # 原始表达式
    expr_str = "x**2 / (1.0 - cos(x))"
    expr = sympify(expr_str)

    # 转换为 AST
    ast = expression_to_ast(expr)
    print("AST:", ast)

    # 从 AST 还原表达式
    recovered_expr = ast_to_expression(ast)
    print("Recovered:", recovered_expr)

    # 检查是否等价
    print("Original == Recovered?", simplify(expr - recovered_expr) == 0)