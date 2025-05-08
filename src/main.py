import sys
import os
from sympy import sympify
from collections import defaultdict

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
expr_parser_dir = os.path.join(current_dir, './expression_parser')
rl_dir = os.path.join(current_dir, './RL')
# 检查路径是否存在
if os.path.exists(expr_parser_dir):
    sys.path.append(expr_parser_dir)
    sys.path.append(rl_dir)
else:
    print(f"指定的路径不存在: {expr_parser_dir}")
    print(f"指定的路径不存在: {rl_dir}")

from parserUtils import encode
from embeddingUtils import *
from similarityUtils import *
from evaluateUtils import *
from actionUtils import *
from Q_RL2 import rl_run


if __name__ == "__main__":
    # 输入表达式，定义域
    expr = "log(1 - x) / log(1 + x)"
    size = 50000
    light = 0.1
    right = 0.9
    # 规范表达式
    expr = str(sympify(expr))
    # 编码
    expr_ast_huffman = encode(expr)
    # 计算嵌入向量
    embedding = get_embedding(expr_ast_huffman)
    # 对比向量库 得到相似表达式列表
    similarity_exprs = get_similarity_exprs(embedding)
    # 强化模块
    reward = rl_run(expr, similarity_exprs,size, light, right)
    if reward > 0 :
        # 存入向量库。
        print(f"表达式优化效果：{reward}  更新向量库")
        save_embeddings(expr, embedding)
    #