from sympy import (
    sympify, Basic, Symbol, Integer, Float, Rational,
    Add, Mul, Pow, Function, cos, sin, tan, exp, log,
    asin, acos, atan, sinh, cosh, tanh, S
)

# 哈夫曼编码表
huffman_code_dict = {
    'AccumulationBounds': '01001110100111',
    'Add': '1100',
    'ComplexInfinity': '01001111',
    'Exp1': '010011100',
    'Float': '0100111010010',
    'Half': '10011',
    'ImaginaryUnit': '0100111011',
    'Infinity': '01001110100000',
    'Integer': '1110',
    'Mul': '101',
    'NaN': '01001110101',
    'NegativeOne': '1101',
    'One': '111100',
    'Pi': '10010',
    'Pow': '011',
    'Rational': '010010',
    'Symbol': '00',
    'Zero': '0100110',
    'acos': '01000',
    'asin': '111110',
    'asinh': '01001110100001',
    'atan': '01010',
    'atanh': '0100111010001',
    'cos': '111111',
    'exp': '10001',
    'log': '10000',
    'sin': '01011',
    'sinh': '01001110100110',
    'tan': '111101'
}
reverse_huffman_dict = {v: k for k, v in huffman_code_dict.items()}

# 表达式 → AST
def expression_to_ast(expr):
    expr = sympify(expr)
    def _to_ast(node):
        if isinstance(node, Basic):
            if node.is_Symbol:
                return ('Symbol', [])
            elif node.is_Integer:
                return ('Integer', int(node))
            elif node.is_Float:
                return ('Float', float(node))
            elif node.is_Rational:
                return ('Rational', (int(node.p), int(node.q)))
            elif node is S.One:
                return ('One', None)
            elif node is S.NegativeOne:
                return ('NegativeOne', None)
            elif node is S.Zero:
                return ('Zero', None)
            elif node is S.Half:
                return ('Half', None)
            elif node is S.Pi:
                return ('Pi', None)
            elif node is S.Exp1:
                return ('Exp1', None)
            elif node is S.ImaginaryUnit:
                return ('ImaginaryUnit', None)
            elif node.is_Add or node.is_Mul or node.is_Pow:
                return (type(node).__name__, [_to_ast(arg) for arg in node.args])
            elif isinstance(node, Function):
                # 处理所有支持的函数（如 sin, cos, exp, log 等）
                func_name = type(node).__name__
                if func_name in huffman_code_dict:
                    return (func_name, [_to_ast(arg) for arg in node.args])
                else:
                    raise ValueError(f"Unsupported function: {func_name}")
            else:
                raise ValueError(f"Unsupported SymPy node: {node}")
        else:
            return ('Literal', node)
    return _to_ast(expr)

# AST → 哈夫曼编码 AST
def ast_to_huffman(ast, huffman_dict):
    node_type, payload = ast
    encoded_type = huffman_dict.get(node_type, '')
    if not encoded_type:
        raise ValueError(f"Unknown node type: {node_type}")
    if isinstance(payload, list):
        return (encoded_type, [ast_to_huffman(child, huffman_dict) for child in payload])
    elif payload is None:
        return (encoded_type, [])
    else:
        return (encoded_type, payload)

# 哈夫曼编码 AST → AST
def huffman_to_ast(huffman_ast, reverse_dict):
    encoded_type, payload = huffman_ast
    if encoded_type in reverse_dict:
        node_type = reverse_dict[encoded_type]
    else:
        raise ValueError(f"Unknown Huffman code: {encoded_type}")
    if isinstance(payload, list) and all(isinstance(child, tuple) for child in payload):
        return (node_type, [huffman_to_ast(child, reverse_dict) for child in payload])
    elif payload == []:
        return (node_type, None)
    else:
        return (node_type, payload)

# AST → 表达式
def ast_to_expression(ast):
    node_type, payload = ast
    if node_type == 'Symbol':
        return Symbol('x')  # 统一符号
    elif node_type == 'Integer':
        return Integer(payload)
    elif node_type == 'Float':
        return Float(payload)
    elif node_type == 'Rational':
        p, q = payload
        return Rational(p, q)
    elif node_type == 'One':
        return S.One
    elif node_type == 'NegativeOne':
        return S.NegativeOne
    elif node_type == 'Zero':
        return S.Zero
    elif node_type == 'Half':
        return S.Half
    elif node_type == 'Pi':
        return S.Pi
    elif node_type == 'Exp1':
        return S.Exp1
    elif node_type == 'ImaginaryUnit':
        return S.ImaginaryUnit
    elif node_type in ('Add', 'Mul', 'Pow'):
        args = [ast_to_expression(child) for child in payload]
        return {'Add': Add, 'Mul': Mul, 'Pow': Pow}[node_type](*args)
    elif node_type in ('sin', 'cos', 'tan', 'exp', 'log', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'):
        args = [ast_to_expression(child) for child in payload]
        return {
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'exp': exp,
            'log': log,
            'asin': asin,
            'acos': acos,
            'atan': atan,
            'sinh': sinh,
            'cosh': cosh,
            'tanh': tanh,
        }[node_type](*args)
    else:
        raise ValueError(f"Unknown node type: {node_type}")

def encode(expr):
    """
    : 对输入的表达式进行编码,返回采用哈夫曼编码后的结果.暂时并没有将其转成二进制的形式进行保存.而是进行哈夫曼编码后,采用索引映射的方式.
    :param expr:
    :return: 编码后的ast
    """
    ast = expression_to_ast(expr)
    huffman_ast = ast_to_huffman(ast, huffman_code_dict)
    return huffman_ast


def decode(huffman_ast):
    """
    : 对输入的编码结果进行解码操作,还原成原始表达式
    :param huffman_ast:
    :return: 还原原始表达式
    """
    recovered_ast = huffman_to_ast(huffman_ast, reverse_huffman_dict)
    recovered_expr = ast_to_expression(recovered_ast)
    return recovered_expr


if __name__ == '__main__':
    expr = "(x / (x + 1))"
    print(expr)
    ast = encode(expr)
    print(ast)
    expr1 =  decode(ast)
    print(expr1)