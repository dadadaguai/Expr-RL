import torch
from torch.nn.functional import cosine_similarity
from embeddingUtils import load_embeddings, get_expr_from_pkl, get_embedding
from parserUtils import encode


class ASTSimilarityBatch:
    def __init__(self, embeddings_path, model=None):
        """
        初始化批量相似度对比器
        :param embeddings_path: 预计算的AST向量文件路径（.pt）
        :param model: 可选的模型（用于实时生成嵌入）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        # 加载预计算向量库
        self.library_vectors = torch.load(embeddings_path, map_location=self.device)
        if self.library_vectors.dim() == 1:
            self.library_vectors = self.library_vectors.unsqueeze(0)

    def query(self, input_vectors, threshold=0.85, top_k=3):
        """
        查询与输入AST最相似的前k个AST
        :param input_vectors: 输入向量或向量对
                            - 单个向量: [hidden_dim] 或 [1, hidden_dim]
                            - 两个向量: [2, hidden_dim]
        :param threshold: 相似度阈值
        :param top_k: 最大返回数量
        :return: 根据输入类型返回：
                - 单输入: List[Tuple[index, embedding, similarity]]
                - 双输入: 相似度分数（如果>=threshold）
        """
        # 输入处理
        if input_vectors.dim() == 1:
            input_vectors = input_vectors.unsqueeze(0)

        # 双输入模式（直接比较两个向量）
        if input_vectors.shape[0] == 2:
            sim = cosine_similarity(input_vectors[0:1], input_vectors[1:2]).item()
            return sim if sim >= threshold else None

        # 单输入模式（查询库）
        sim_scores = cosine_similarity(
            input_vectors.unsqueeze(1),
            self.library_vectors.unsqueeze(0),
            dim=2
        ).squeeze(0)

        # 筛选结果
        mask = sim_scores >= threshold
        valid_scores = sim_scores[mask]
        valid_indices = torch.nonzero(mask).flatten()

        # 包装结果
        results = []
        if len(valid_scores) > 0:
            # 获取top-k结果
            top_scores, top_indices = torch.topk(valid_scores, k=min(top_k, len(valid_scores)))

            for score, idx in zip(top_scores, valid_indices[top_indices]):
                results.append((
                    idx.item(),  # 索引
                    self.library_vectors[idx].cpu(),  # 嵌入向量
                    score.item()  # 相似度
                ))

        return results

    def get_embedding(self, ast):
        """使用模型生成AST的嵌入向量"""
        if self.model is None:
            raise ValueError("Model not provided for embedding generation")
        with torch.no_grad():
            return self.model.tree_lstm(ast).to(self.device)



def get_similarity_exprs(expr_embedding):
    embeddings_path = "../data/embeddings.pt"
    import os
    if os.path.exists(embeddings_path):
        existing_embeddings = torch.load(embeddings_path)
    else:
        print("文件不存在")
    sim_checker = ASTSimilarityBatch(embeddings_path="../data/embeddings.pt")

    matches = sim_checker.query(expr_embedding)
    expr_list  = []

    print("匹配结果：")
    for idx, emb, score in matches:
        print(f"索引: {idx}, 相似度: {score:.4f}")
        print(f"嵌入向量: {emb[:5]}...")  # 只打印前5维
        matches_expr =  get_expr_from_pkl(emb)
        print(f"匹配的表达式: {matches_expr}")
        expr_list.append(matches_expr)
    print(expr_list)
    return expr_list

def test_get_similarity_exprs(expr):
    ast_huffman = encode(expr)
    expr_embedding = get_embedding(ast_huffman)

    embeddings_path = "../data/embeddings.pt"
    import os
    if os.path.exists(embeddings_path):
        existing_embeddings = torch.load(embeddings_path)
    else:
        print("文件不存在")
    sim_checker = ASTSimilarityBatch(embeddings_path="../data/embeddings.pt")

    matches = sim_checker.query(expr_embedding)
    expr_list  = []

    print("匹配结果：")
    for idx, emb, score in matches:
        print(f"索引: {idx}, 相似度: {score:.4f}")
        print(f"嵌入向量: {emb[:5]}...")  # 只打印前5维
        matches_expr =  get_expr_from_pkl(emb)
        print(f"匹配的表达式: {matches_expr}")
        expr_list.append(matches_expr)
    print(expr_list)
    return expr_list

# 使用示例
if __name__ == "__main__":
    expr = "(x / (x + 1))"
    get_similarity_exprs(expr)
    # 初始化（只需要嵌入文件）
    # sim_checker = ASTSimilarityBatch(embeddings_path="embeddings.pt")
    # embedding1 = load_embeddings('embeddings.pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #
    # matches = sim_checker.query(embedding1)
    #
    # print("匹配结果：")
    # for idx, emb, score in matches:
    #     print(f"索引: {idx}, 相似度: {score:.4f}")
    #     print(f"嵌入向量: {emb[:5]}...")  # 只打印前5维
    #     print(f"匹配的表达式: {get_expr_from_pkl(emb)}")


    # 示例2：双输入比较
    # emb1 = torch.randn(128)
    # emb2 = torch.randn(128)
    # similarity = sim_checker.query(torch.stack([emb1, emb2]))
    # print(f"相似度: {similarity if similarity is not None else '低于阈值'}")

    # 示例3：使用模型实时生成嵌入
    # sim_checker = ASTSimilarityBatch(embeddings_path="embeddings.pt", model=your_model)
    # ast = get_some_ast()  # 获取AST结构
    # embedding = sim_checker.get_embedding(ast)
    # matches = sim_checker.query(embedding)