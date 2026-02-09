# Expr-RL: 基于强化学习的表达式精度优化框架

## 项目简介
Expr-RL 是一个基于强化学习（Q-Learning）的数学表达式重写与精度优化框架，旨在通过自动重写数学表达式来降低浮点计算中的数值误差（ULP误差）。该框架能够针对包含对数、指数、三角函数等复杂表达式，自动选择最优的重写策略，提升数值计算的稳定性和精度。

## 核心功能
- 表达式AST编码与嵌入向量生成
- 基于余弦相似度的相似表达式匹配
- 多种表达式重写策略（分配律、结合律、log1p/expm1优化、sqrt共轭化简等）
- 基于Q-Learning的强化学习优化
- 浮点误差评估（ULP、相对误差、Herbie风格误差）
- 滑动窗口采样与误差分析

## 项目结构
```
Expr-RL/
├── src/
│   ├── main.py                 # 主程序入口
│   ├── expression_parser/      # 表达式解析模块
│   │   ├── parserUtils.py      # AST编码工具
│   │   ├── embeddingUtils.py   # 嵌入向量生成
│   │   └── similarityUtils.py  # 相似度匹配
│   └── RL/                     # 强化学习核心模块
│       ├── evaluateUtils.py    # 精度评估工具
│       ├── actionUtils.py      # 表达式重写动作
│       ├── Q_RL.py             # Q-Learning实现
│       └── Q_RL2.py            # RL运行入口
├── data/
│   └── embeddings.pt           # 预训练表达式嵌入向量库
└── requirements.txt            # 依赖包列表
```

## 核心模块详解

### 1. 表达式解析模块 (expression_parser)
#### parserUtils.py
- 提供表达式AST的哈夫曼编码功能
- 将数学表达式转换为结构化的AST表示

#### embeddingUtils.py
- 生成AST的嵌入向量
- 向量归一化与存储

#### similarityUtils.py
- 基于余弦相似度匹配相似表达式
- 从预训练向量库中检索最优匹配

### 2. 强化学习核心模块 (RL)
#### evaluateUtils.py
- **误差计算**：ULP误差、相对误差、Herbie风格误差
- **采样方法**：均匀采样、滑动窗口采样
- **精度评估**：64位/128位精度对比计算
- **误差分析**：最大误差点检测、聚类分析

#### actionUtils.py
- 提供多种表达式重写策略：
  - 基础化简：expand、factor、cancel、simplify
  - 特殊函数优化：log1p、expm1、log(exp)化简
  - 代数变换：分配律、结合律、共轭化简
  - 三角函数优化：expand_trig、trigsimp

#### Q_RL.py
- Q-Learning智能体实现
- 表达式优化环境定义
- 奖励函数设计（基于精度提升）
- Q表的保存与加载

## 环境准备

### 依赖安装
```bash
# 克隆项目
git clone https://github.com/dadadaguai/Expr-RL.git
cd Expr-RL

# 安装依赖（建议使用虚拟环境）
# 核心依赖包括：
# - sympy: 表达式解析与重写
# - numpy: 数值计算
# - scikit-learn: 聚类与相似度计算
# - gmpy2: 高精度计算
# - torch: 嵌入向量计算
# - pandas: 数据处理
```

### 依赖包说明
| 包名 | 用途 |
|------|------|
| sympy | 数学表达式解析、重写、化简 |
| numpy | 数值计算、采样、数组处理 |
| gmpy2 | 高精度浮点运算、误差计算 |
| scikit-learn | 余弦相似度、聚类分析 |
| torch | 嵌入向量计算与相似度匹配 |
| pandas | 数据存储与分析 |
| csv | 误差数据导出 |

## 使用指南

### 基础使用（快速开始）
1. 修改`main.py`中的配置参数：
```python
if __name__ == "__main__":
    # 输入表达式，定义域
    expr = "log(1 - x) / log(1 + x)"  # 需要优化的表达式
    size = 50000                      # 采样点数量
    light = 0.1                       # 定义域左边界
    right = 0.9                       # 定义域右边界
    # ... 其余代码保持不变
```

2. 运行主程序：
```bash
cd src
python main.py
```

3. 输出说明：
   - 相似表达式匹配结果
   - 强化学习优化过程
   - 精度提升百分比
   - 优化后的表达式
   - （可选）更新嵌入向量库

### 高级使用

#### 1. 自定义表达式优化
```python
from RL.Q_RL import ExpressionEnvironment, QLearningAgent, train_agent
from sympy import symbols

# 定义优化目标
x = symbols('x')
expr = "log(1+x) - x/(1+x)"  # 待优化表达式

# 创建环境和智能体
env = ExpressionEnvironment(expr)
agent = QLearningAgent(env.actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.01)

# 训练智能体
train_agent(env, agent, expression_id="custom_expr", episodes=1000)

# 优化表达式
optimized_expr = env.current_expr
print(f"优化后的表达式: {optimized_expr}")
```

#### 2. 误差分析工具使用
```python
from RL.evaluateUtils import oneParaErrorDetect, uniform_sampling

# 定义表达式和采样范围
expr = "log(1 - x) / log(1 + x)"
samples = uniform_sampling(0.1, 0.9, 1000)  # 生成1000个采样点

# 计算每个采样点的ULP误差
errors = [oneParaErrorDetect(expr, x) for x in samples]

# 分析最大误差
max_error = max(errors)
max_error_idx = errors.index(max_error)
print(f"最大ULP误差: {max_error} 在 x = {samples[max_error_idx]}")
```

#### 3. 表达式重写策略单独使用
```python
from RL.actionUtils import optimize_log1p, smart_rewrite
from sympy import sympify

# 原始表达式
expr = sympify("log(x)")

# 使用log1p优化
optimized_expr = optimize_log1p(expr)
print(f"优化前: {expr}")
print(f"优化后: {optimized_expr}")  # 输出: log1p(x - 1)

# 智能重写
expr2 = sympify("x/(x+1)")
rewritten_expr = smart_rewrite(expr2)
print(f"智能重写后: {rewritten_expr}")  # 输出: 1 - 1/(x + 1)
```

### 配置说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| size | 采样点数量 | 50000 |
| light | 定义域左边界 | 0.1 |
| right | 定义域右边界 | 0.9 |
| learning_rate | Q-Learning学习率 | 0.1 |
| discount_factor | 折扣因子 | 0.99 |
| epsilon | 探索率 | 0.01 |
| episodes | 训练轮数 | 1000 |
| window_size | 滑动窗口大小 | 400 |
| step | 滑动窗口步长 | 100 |
| ULP | ULP误差阈值 | 1.0 |

## 关键功能说明

### 1. 误差评估指标
- **ULP误差**：单位最后位置误差，衡量浮点表示的精度
- **相对误差**：(近似值-精确值)/精确值
- **Herbie风格误差**：以2为底的对数误差度量

### 2. 表达式重写策略
| 策略 | 适用场景 | 优化效果 |
|------|----------|----------|
| log1p优化 | log(x) 其中x接近1 | 减少小数值误差 |
| expm1优化 | exp(x)-1 其中x接近0 | 减少小数值误差 |
| 共轭化简 | sqrt(a)±sqrt(b) | 避免数值抵消 |
| 智能分式重写 | x/(x+c) | 转换为1 - c/(x+c) |
| 分配律/结合律 | 复杂乘加表达式 | 减少计算步骤 |

### 3. 强化学习优化流程
1. 初始化表达式环境，识别可用的重写动作
2. Q-Learning智能体选择动作（探索/利用）
3. 执行表达式重写，计算精度提升奖励
4. 更新Q表，优化动作选择策略
5. 迭代直至收敛或达到最大步数
6. 返回最优表达式

## 注意事项
1. **定义域设置**：确保表达式在指定定义域内有定义，避免除零、对数负数等错误
2. **采样点数量**：数量越大精度越高，但计算时间越长
3. **强化学习参数**：
   - 学习率过高可能导致训练不稳定
   - 探索率过小可能陷入局部最优
4. **精度计算**：使用gmpy2的128位高精度计算作为参考值
5. **嵌入向量库**：首次使用需要生成或加载预训练的embeddings.pt文件

## 扩展开发
1. **新增重写策略**：在`actionUtils.py`中添加自定义重写函数，更新`classification_function`
2. **自定义奖励函数**：修改`Q_RL.py`中的`evaluate_precision`函数
3. **新增误差指标**：在`evaluateUtils.py`中添加新的误差计算函数
4. **扩展表达式类型**：支持更多特殊函数（如双曲函数、贝塞尔函数等）

## 常见问题
1. **导入错误**：确保所有路径添加到sys.path，检查依赖包版本
2. **精度计算错误**：检查表达式定义域，避免无穷大/NaN值
3. **Q表加载失败**：首次运行需先训练生成Q表
4. **相似度匹配为空**：确保embeddings.pt文件存在且包含相似表达式

## 示例输出
```
匹配结果：
索引: 5, 相似度: 0.9234
嵌入向量: [0.123, 0.456, 0.789, 0.012, 0.345]...
匹配的表达式: log(1+x)/x

Episode 1:
Step 1: Action: optimize_log1p, Reward: 15.23
Step 2: Action: smart_rewrite, Reward: 8.76
Episode 1 completed. Total reward: 24.00

表达式优化效果：24.00  更新向量库
优化后的表达式: log1p(x)/(1 + x)
```
