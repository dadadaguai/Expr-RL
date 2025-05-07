import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from sympy import *
import os
import pickle


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

class HuffmanEmbedding(nn.Module):
    def __init__(self, huffman_code_dict, embedding_dim):
        super().__init__()
        self.huffman_code_dict = huffman_code_dict
        self.embedding_dim = embedding_dim
        # 创建可训练的嵌入层，每个字符（'0'或'1'）都有一个嵌入向量
        self.bit_embedding = nn.Embedding(2, embedding_dim // 2)  # 每个bit用embedding_dim//2维表示
        self.projection = nn.Linear(embedding_dim // 2 * max(len(code) for code in huffman_code_dict.values()),
                                    embedding_dim)

    def forward(self, huffman_code):
        # 将哈夫曼编码字符串转换为bit序列
        bits = [int(bit) for bit in huffman_code]
        bits_tensor = torch.tensor(bits, dtype=torch.long).to(self.bit_embedding.weight.device)  # 确保张量在正确设备上

        # 获取每个bit的嵌入
        bit_embeddings = self.bit_embedding(bits_tensor)

        # 如果编码长度不足最大长度，填充零
        max_len = max(len(code) for code in self.huffman_code_dict.values())
        if len(bits) < max_len:
            padding = torch.zeros(max_len - len(bits), self.embedding_dim // 2,
                                  device=self.bit_embedding.weight.device)  # 确保填充张量在正确设备上
            bit_embeddings = torch.cat([bit_embeddings, padding], dim=0)

        # 展平并投影到最终嵌入维度
        flattened = bit_embeddings.view(-1)
        return self.projection(flattened)


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
    def __init__(self, huffman_code_dict, embedding_dim, hidden_dim):
        super().__init__()
        self.huffman_embedding = HuffmanEmbedding(huffman_code_dict, embedding_dim)
        self.lstm_cell = NaryTreeLSTMCell(embedding_dim, hidden_dim)

    def forward(self, huffman_ast):
        device = next(self.parameters()).device  # 获取模型所在设备

        def _traverse(node):
            huffman_code, children = node
            x = self.huffman_embedding(huffman_code)

            # 检查是否是 Integer 节点（哈夫曼编码为 '1110'）
            if huffman_code == '1110' and isinstance(children, int):
                # 如果是 Integer 节点，且 children 是整数，则视为叶子节点
                return self.lstm_cell(x, [])
            elif not children or isinstance(children, int):
                # 其他叶子节点或无效结构
                return self.lstm_cell(x, [])
            else:
                # 正常内部节点
                children_states = [_traverse(child) for child in children]
                return self.lstm_cell(x, children_states)

        h_root, _ = _traverse(huffman_ast)
        return h_root


class ASTSimilarity(nn.Module):
    def __init__(self, huffman_code_dict, embedding_dim, hidden_dim):
        super().__init__()
        self.tree_lstm = TreeLSTM(huffman_code_dict, embedding_dim, hidden_dim)

    def forward(self, ast1, ast2):
        h1 = self.tree_lstm(ast1)
        h2 = self.tree_lstm(ast2)
        return cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))


def get_embedding(huffman_ast):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    model = ASTSimilarity(huffman_code_dict, 64, 128).to(device)

    # 加载模型权重
    # huffman_ast = ('101', [('00', []), ('011', [('1100', [('1110', 1), ('00', [])]), ('1110', -1)])])
    # model.load_state_dict(torch.load('ast_similarity_model.pth', map_location=device))
    embedding = model.tree_lstm(huffman_ast)
    return embedding


def save_embeddings(expr, embedding, path_em='embeddings.pt', path_kv='em_expr.pickle'):
    embedding_cpu = embedding.cpu()  # 确保张量在CPU上

    # 保存嵌入向量到 .pt 文件
    if os.path.exists(path_em):
        existing_embeddings = torch.load(path_em).cpu()  # 确保现有的嵌入向量在CPU上
        new_embeddings = torch.cat((existing_embeddings, embedding_cpu.unsqueeze(0)), dim=0)
        torch.save(new_embeddings, path_em)
    else:
        torch.save(embedding_cpu.unsqueeze(0), path_em)

    # 保存（嵌入向量，表达式）作为键值对到 .pickle 文件
    kv_pair = (str(embedding_cpu.detach().numpy().tolist()), expr)
    print(kv_pair)
    if os.path.exists(path_kv):
        with open(path_kv, 'rb') as file:
            existing_kv = pickle.load(file)
        # 检查键是否已存在，如果不存在则追加
        if kv_pair[0] not in existing_kv:
            existing_kv[kv_pair[0]] = kv_pair[1]  # 使用新的键值对更新字典

    else:
        existing_kv = {kv_pair[0]: kv_pair[1]}  # 初始化一个新的字典来保存键值对
    with open(path_kv, 'wb') as file:
        pickle.dump(existing_kv, file)

def load_embeddings(path_em='embeddings.pt'):
    if os.path.exists(path_em):
        existing_embeddings = torch.load(path_em).cpu()
        # 如果需要一维数组，可以在这里使用 .squeeze(0)
        return existing_embeddings.squeeze(0) if existing_embeddings.dim() == 2 and existing_embeddings.size(
            0) == 1 else existing_embeddings


def load_embeddings_expr(path_em='em_expr.npy'):
    pass

# 根据嵌入向量返回expr,用于经验库的匹配
def get_expr_from_pkl(embedding,path_kv='em_expr.pickle'):
    # 加载pickle文件中的字典
    if os.path.exists(path_kv):
        with open(path_kv, 'rb') as file:
            kv_dict = pickle.load(file)
    else:
        return None  # 如果文件不存在，返回 None
    k_embedding = str(embedding.detach().numpy().tolist())
    if k_embedding not in kv_dict:
        return None
    expr = kv_dict[k_embedding]
    return expr

if __name__ == '__main__':
    pass
    #
    # # 加载模型权重
    # huffman_ast = ('101', [('00', []), ('011', [('1100', [('1110', 1), ('00', [])]), ('1110', -1)])])
    # embedding = get_embedding(huffman_ast)
    # print(embedding)
    # save_embeddings('(x / (x + 1))', embedding)
    #
    # embedding1 = load_embeddings('embeddings.pt')
    # print(embedding1)
    # # get_expr_from_pkl(embedding1)
    # embedding2 = embedding1.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # sim = cosine_similarity(embedding, embedding2, dim=1)
    # print(f"sim: {sim}")


