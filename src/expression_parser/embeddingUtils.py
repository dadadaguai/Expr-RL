import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from sympy import *
from data.预编码的AST列表 import get_ast_library, encode_ast
from parser import encode

class NaryTreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 门控参数
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f_left = nn.Linear(hidden_dim, hidden_dim)
        self.U_f_right = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, children_states):
        # 初始化子节点状态（与x同设备）
        zero_state = torch.zeros(self.hidden_dim, device=x.device)
        h_left, c_left = children_states[0] if len(children_states) > 0 else (zero_state, zero_state)
        h_right, c_right = children_states[1] if len(children_states) > 1 else (zero_state, zero_state)

        # 计算门控
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_left + h_right))
        f_left = torch.sigmoid(self.W_f(x) + self.U_f_left(h_left))
        f_right = torch.sigmoid(self.W_f(x) + self.U_f_right(h_right))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_left + h_right))
        u = torch.tanh(self.W_u(x) + self.U_u(h_left + h_right))

        # 更新状态
        c = i * u + f_left * c_left + f_right * c_right
        h = o * torch.tanh(c)
        return h, c


class TreeLSTM(nn.Module):
    def __init__(self, huffman_code_to_index, embedding_dim, hidden_dim):
        super().__init__()
        self.huffman_code_to_index = huffman_code_to_index
        self.vocab_size = len(huffman_code_to_index)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm_cell = NaryTreeLSTMCell(embedding_dim, hidden_dim)

    def forward(self, ast):
        device = next(self.parameters()).device  # 获取模型所在设备

        def _traverse(node):
            node_type, children = node
            index = torch.tensor(self.huffman_code_to_index[node_type], device=device)
            x = self.embedding(index).squeeze(0)
            if not children:  # 叶子节点
                return self.lstm_cell(x, [])
            else:  # 内部节点
                children_states = [_traverse(child) for child in children]
                return self.lstm_cell(x, children_states)

        h_root, _ = _traverse(ast)
        return h_root

class ASTSimilarity(nn.Module):
    def __init__(self, huffman_code_to_index, embedding_dim, hidden_dim):
        super().__init__()
        self.tree_lstm = TreeLSTM(huffman_code_to_index, embedding_dim, hidden_dim)

    def forward(self, ast1, ast2):
        h1 = self.tree_lstm(ast1)
        h2 = self.tree_lstm(ast2)
        return cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))

# vocab_size = len(vocab)
# # 初始化模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ASTSimilarity(vocab_size, 64, 128).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # 保存模型权重
# torch.save(model.state_dict(), 'ast_similarity_model.pth')

def get_ast_embedding():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 假设 vocab_size 已知，这里需要根据实际情况设置
    import pickle
    # 读取词汇表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print(vocab)
    # 读取哈夫曼编码
    # with open('huffman_code.pkl', 'rb') as f:
    #     huffman_code = pickle.load(f)
    size = len(vocab)  # 示例值，根据你的词汇表大小调整
    # vocab = {sym: i for i, (sym, code) in enumerate(huffman_code)}
    # 初始化模型
    model = ASTSimilarity(size, 64, 128).to(device)
    # 加载模型权重
    model.load_state_dict(torch.load('ast_similarity_model.pth', map_location=device))
    ast_code = encode("(x / (x + 1))")
    embeddings = model.tree_lstm()
    print(embeddings)
    return embeddings
    # embedding, asts = precompute_embeddings(model, pre_encoded_asts)

    print(embeddings)
def precompute_embeddings(model, pre_encoded_asts, device='cuda'):
    """
    预计算所有AST的嵌入向量
    :param vocab: 符号到索引的映射表
    :param device: 计算设备
    :return: 嵌入向量矩阵 [num_asts, hidden_dim], 对应的AST列表
    """
    model.eval()
    model.to(device)

    # 编码AST并存储向量
    ast_vectors = []
    valid_asts = []
    for ast_encoded in pre_encoded_asts:
        with torch.no_grad():
            vector = model.tree_lstm(ast_encoded).to('cpu')  # 向量计算并移到CPU
            ast_vectors.append(vector)
            valid_asts.append(ast_encoded)

    return torch.stack(ast_vectors), valid_asts

def get_ast_encoder(ast):
    import pickle

    # 读取词汇表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    ast_encode = encode_ast(ast, vocab)
    return ast_encode

def expression_to_ast_with_symbols(expression):
    """
    将表达式转换为包含符号类型的 AST 节点信息。
    返回的是一个以 (节点类型, 子节点) 形式构成的嵌套结构。
    """
    expr = sympify(expression)

    def build_ast(node):
        if isinstance(node, Basic):
            return (type(node).__name__, [build_ast(arg) for arg in node.args])
        else:
            return repr(node)  # 如果不是SymPy类型，直接返回其表示

    return build_ast(expr)



if __name__ == '__main__':
    get_ast_embedding()
# 加载模型权重
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 假设 vocab_size 已知，这里需要根据实际情况设置
#     import pickle
#     # 读取词汇表
#     with open('vocab.pkl', 'rb') as f:
#         vocab = pickle.load(f)
#     print(vocab)
#     # 读取哈夫曼编码
#     # with open('huffman_code.pkl', 'rb') as f:
#     #     huffman_code = pickle.load(f)
#     size = len(vocab)  # 示例值，根据你的词汇表大小调整
#     # vocab = {sym: i for i, (sym, code) in enumerate(huffman_code)}
#     # 初始化模型
#     model = ASTSimilarity(size, 64, 128).to(device)
#     # 加载模型权重
#     model.load_state_dict(torch.load('ast_similarity_model.pth', map_location=device))
#     pre_encoded_asts = get_ast_library()
#     embeddings, asts = precompute_embeddings(model, pre_encoded_asts)
#
#     print(embeddings)
#     print(asts)
#     torch.save(embeddings, 'ast_embeddings.pt')  # 保存向量

    # # 保存词汇表（用于后续编码）
    # import pickle
    # with open('vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    #
    # # 可选：保存整个模型结构（不推荐，可能导致兼容性问题）
    # # torch.save(model, 'full_model.pt')