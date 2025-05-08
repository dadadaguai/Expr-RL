import random
from collections import defaultdict

import numpy as np
from sympy import (sympify, expand, factor, cancel, simplify,
                   expand_trig, trigsimp, expand_log, logcombine,
                   expand_power_base, powsimp, powdenest, together,
                   fraction, symbols, apart, log, preorder_traversal, Function, exp)
import re
import pickle

from config import size,light,right
from evaluateUtils import get_expr_res, cache_evaluate
from actionUtils import custom_distribute, optimize_log1p, log_exp_replace, optimize_expm1, \
    sqrt_rewrite, smart_rewrite, associative_law


import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造 expression_parser 目录的路径
expr_parser_dir = os.path.join(current_dir, '../../src/expression_parser')

# 检查路径是否存在
if os.path.exists(expr_parser_dir):
    # 将 expression_parser 目录添加到 sys.path
    sys.path.append(expr_parser_dir)
else:
    print(f"指定的路径不存在: {expr_parser_dir}")

from similarityUtils import get_similarity_exprs


# 定义符号
x = symbols('x')
oracle = None
ulp_cache = {}
strategy_library = defaultdict(list)  # 策略库，保存成功的策略序列
# 精度优化函数
def evaluate_precision(current_expr, new_expr, size, light, right):
    """
    评估表达式的精度优化效果。
    返回一个分数，表示新表达式相对于旧表达式的精度优化程度。
    """
    if len(oracle_cache) == 1 :
        oracle = oracle_cache[0]

    if str(sympify(current_expr)) in ulp_cache:
        current = ulp_cache[str(sympify(current_expr))]
    else:
        current = cache_evaluate(current_expr, oracle, size, light, right)
        ulp_cache[str(sympify(current_expr))] = current

    if str(sympify(new_expr)) in ulp_cache:
        optimized = ulp_cache[str(sympify(new_expr))]
    else:
        optimized = cache_evaluate(new_expr, oracle, size, light, right)
        ulp_cache[str(sympify(new_expr))] = optimized

    avg_ulp_origin = np.mean(current[:, 1])
    avg_ulp_res = np.mean(optimized[:, 1])
    ulp_improvement = avg_ulp_origin - avg_ulp_res
    ulp_improvement_percentage = (ulp_improvement / avg_ulp_origin) * 100
    return ulp_improvement_percentage


class StateManager:
    def __init__(self):
        self.state_history = []
        self.action_history = []
        self.current_state = None
        self.current_optimization = 0.0

    def update(self, new_state, action, optimization_value=0.0):
        self.state_history.append((self.current_state, self.current_optimization))
        self.action_history.append(action)
        self.current_state = new_state
        self.current_optimization = optimization_value

    def rollback(self):
        if len(self.state_history) > 0:
            prev_state, prev_optimization = self.state_history.pop()
            self.current_state = prev_state
            self.current_optimization = prev_optimization
            return self.action_history.pop()
        return None

    def get_last_actions(self, n=3):
        return self.action_history[-n:] if len(self.action_history) >= n else None

    def reset(self, initial_state, initial_optimization=0.0):
        self.state_history = []
        self.action_history = []
        self.current_state = initial_state
        self.current_optimization = initial_optimization


class ExpressionEnvironment:
    def __init__(self, initial_expr,size,light,right):
        self.initial_expr = initial_expr
        self.current_expr = sympify(initial_expr)
        self.actions = self.determine_actions(initial_expr)
        self.state_manager = StateManager()
        self.state_manager.reset(str(self.current_expr))
        self.size = size
        self.light = light
        self.right = right

    def determine_actions(self, expr):
        actions = set([associative_law, custom_distribute, smart_rewrite, expand, factor, cancel])
        math_functions = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'exp', 'log', 'sqrt', 'cbrt', 'pow'}
        pattern = r'\b(' + '|'.join(math_functions) + r')\b'
        found_functions = set(re.findall(pattern, expr))
        for func in found_functions:
            if func in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
                actions.update([expand_trig, trigsimp])
            elif func in ['log']:
                actions.update([log_exp_replace, expand_log, logcombine, optimize_log1p])
            elif func in ['exp']:
                actions.update([optimize_expm1, log_exp_replace])
            elif func in ['sqrt']:
                sqrt_pattern = r"sqrt\(([^)]+)\)"
                matches = re.findall(sqrt_pattern, str(expr))
                if len(matches) == 2:
                    actions.add(sqrt_rewrite)
            elif func in ['pow']:
                actions.update([expand_power_base, powsimp, powdenest])
        return list(actions)

    def reset(self, initial_expr=None):
        if initial_expr is not None:
            self.initial_expr = initial_expr
        self.current_expr = sympify(self.initial_expr)
        self.state_manager.reset(str(self.current_expr), 0.0)
        return self.current_expr

    def step(self, action):
        old_expr = self.current_expr
        new_exprs = action(self.current_expr)
        if not isinstance(new_exprs, list):
            new_exprs = [new_exprs]
        best_expr = max(new_exprs, key=lambda expr: evaluate_precision(old_expr, expr,self.size,self.light,self.right))
        changed = (old_expr != best_expr)
        if not changed:
            reward = -1
        else:
            optimization_value = evaluate_precision(old_expr, best_expr,size,light,right)
            reward = optimization_value + self.state_manager.current_optimization if optimization_value > 0 else optimization_value
        if changed and reward >= 0:
            self.current_expr = best_expr
            self.state_manager.update(str(self.current_expr), action.__name__, optimization_value)
        return str(self.current_expr), reward, changed


class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.01):
        self.actions = actions
        self.q_tables = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.last_action = None
        self.strategy_memory = defaultdict(list)  # 使用defaultdict自动处理新键

    def get_q_table(self, expression_id):
        if expression_id not in self.q_tables:
            self.q_tables[expression_id] = {}
        return self.q_tables[expression_id]

    def get_q_value(self, expression_id, state, action):
        q_table = self.get_q_table(expression_id)
        return q_table.get((state, action.__name__), 0)

    def update_q_value(self, expression_id, state, action, reward, next_state):
        q_table = self.get_q_table(expression_id)
        available_actions = [a for a in self.actions if a.__name__ != action.__name__]
        if not available_actions:
            available_actions = self.actions

        max_q_next = max([self.get_q_value(expression_id, next_state, a) for a in available_actions], default=0)

        old_q = self.get_q_value(expression_id, state, action)
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_q_next - old_q)
        q_table[(state, action.__name__)] = new_q

    def remember_strategy(self, expr, action_sequence, reward):
        """记录成功的策略序列"""
        expr_key = str(sympify(expr))  # 确保键是规范化的表达式字符串
        self.strategy_memory[expr_key].append((action_sequence, reward))
        # 按奖励排序保存前5个最佳策略
        self.strategy_memory[expr_key] = sorted(self.strategy_memory[expr_key],
                                                key=lambda x: x[1], reverse=True)[:5]

    def choose_action(self, expression_id, state):
        q_table = self.get_q_table(expression_id)
        available_actions = [a for a in self.actions if a.__name__ != self.last_action]
        if not available_actions:
            available_actions = self.actions

        if random.random() < self.epsilon:
            chosen_action = random.choice(available_actions)
        else:
            q_values = [(a, self.get_q_value(expression_id, state, a)) for a in available_actions]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [a for a, q in q_values if q == max_q]
            chosen_action = random.choice(best_actions)

        self.last_action = chosen_action.__name__
        return chosen_action


def train_agent(env, agent, expression_id, episodes=200):
    """强化学习训练函数"""
    print(f"\n开始训练表达式: {expression_id}")
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 20:
            action = agent.choose_action(expression_id, state)
            next_state, reward, changed = env.step(action)

            agent.update_q_value(expression_id, state, action, reward, next_state)

            episode_reward += reward
            state = next_state
            steps += 1

            if reward < 0 or not changed:
                done = True

        # 每50个episode打印一次进度
        if (episode + 1) % 50 == 0:
            print(f"训练进度: {episode + 1}/{episodes} episodes, 最近奖励: {episode_reward:.2f}")



def evaluate_agent(env, agent, expression_id, episodes=100):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        print(f"\nEvaluation Episode {episode + 1}:")
        while not done:
            action = agent.choose_action(expression_id, state)
            state, reward = env.step(action)
            episode_reward += reward
            print(f"Action: {action.__name__}, New Expression: {state}, Reward: {reward}")
            if reward < 0:
                done = True
        total_rewards += episode_reward
    return total_rewards / episodes


def save_q_table(q_tables, filename="./RL/q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(q_tables, f)
    print(f"Q表已保存到 {filename}")


def load_q_table(filename="./RL/q_table.pkl"):
    try:
        with open(filename, "rb") as f:
            loaded_data = pickle.load(f)
        print(f"Q表已从 {filename} 加载")

        # Ensure strategy_memory is a defaultdict
        if 'strategy_memory' in loaded_data:
            strategy_memory = defaultdict(list)
            strategy_memory.update(loaded_data['strategy_memory'])
            loaded_data['strategy_memory'] = strategy_memory

        return loaded_data
    except FileNotFoundError:
        print(f"未找到文件 {filename}，将使用空Q表")
        return {'q_tables': {}, 'strategy_memory': defaultdict(list)}


def get_best_strategy_from_similar(agent, similar_exprs):
    """从相似表达式中获取最佳策略"""
    if not similar_exprs:
        return None
    best_strategy = None
    max_reward = -float('inf')

    for expr in similar_exprs:
        expr = str(sympify(expr))
        if expr in agent.strategy_memory:
            for strategy, reward in agent.strategy_memory[expr]:
                if reward > max_reward:
                    max_reward = reward
                    best_strategy = strategy

    return best_strategy, max_reward


def apply_strategy_sequence(env, strategy_sequence):
    """直接应用策略序列"""
    best_expr = env.current_expr
    best_reward = 0
    applied_actions = []

    for action_name in strategy_sequence:
        # 找到对应的action对象
        action = next((a for a in env.actions if a.__name__ == action_name), None)
        if not action:
            continue

        new_expr, reward, changed = env.step(action)
        if changed and reward > 0:
            applied_actions.append(action_name)
            best_expr = new_expr
            best_reward += reward
        else:
            break

    return str(best_expr), best_reward, applied_actions


def hybrid_optimize(agent, env, expr,similar_exprs):
    """混合优化策略"""
    # 规范表达式ID
    expr_id = str(sympify(expr))
    # 首先尝试从相似表达式中获取最佳策略
    best_strategy, strategy_reward = get_best_strategy_from_similar(agent, similar_exprs)

    if best_strategy:
        print(f"\n找到相似表达式的策略: {best_strategy}, 历史奖励: {strategy_reward:.2f}")
        env.reset(expr_id)
        optimized_expr, reward, actions = apply_strategy_sequence(env, best_strategy)

        if reward > 0:
            print(f"直接应用相似策略成功，奖励: {reward:.2f}")
            agent.remember_strategy(expr_id, actions, reward)
            return optimized_expr, reward, actions
        else:
            print("相似策略无效，将启动强化学习训练...")

    # 如果没有相似策略或相似策略无效，则启动强化学习训练
    print("\n启动强化学习训练寻找新策略...")
    train_agent(env, agent, expr_id, episodes=500)

    # 使用训练后的agent进行优化
    env.reset(expr)
    optimized_expr, reward, actions = optimize_expression(agent, env, expr_id, expr)

    # 保存有效的策略
    if reward > 0:
        agent.remember_strategy(expr_id, actions, reward)
        # 保存更新后的Q表和策略库
        save_data = {
            'q_tables': agent.q_tables,
            'strategy_memory': dict(agent.strategy_memory)  # Convert to regular dict for saving
        }
        save_q_table(save_data)
        print("新策略已保存到Q表和策略库")

    return optimized_expr, reward, actions

def optimize_expression(agent, env, expression_id, expression):
    """使用训练后的agent进行优化"""
    state = env.reset(expression)
    best_expr = state
    best_reward = float('-inf')
    action_sequence = []
    steps = 0
    max_steps = 15

    print("\n开始应用强化学习策略优化...")
    while steps < max_steps:
        action = agent.choose_action(expression_id, state)
        next_state, reward, changed = env.step(action)

        print(f"步骤 {steps + 1}: 动作: {action.__name__}, 奖励: {reward:.2f}, 状态: {next_state}")

        if reward > best_reward:
            best_expr = next_state
            best_reward = reward

        if reward >= 0 and changed:
            action_sequence.append(action.__name__)
        else:
            if len(action_sequence) >= 3:  # 如果有足够多的有效动作，可以提前停止
                break

        state = next_state
        steps += 1

    return best_expr, best_reward, action_sequence
oracle_cache = []
def rl_run(initial_expr,similar_exprs,size,light,right):
    oracle_cache.append(get_expr_res(initial_expr, size, light, right, 113))
    env = ExpressionEnvironment(initial_expr,size, light, right)
    agent = QLearningAgent(env.actions, learning_rate=0.2, discount_factor=0.95, epsilon=0.15)
    expression_id = initial_expr
    # 尝试加载现有的 Q 表和策略库
    loaded_data = load_q_table()
    if loaded_data:
        agent.q_tables = loaded_data.get('q_tables', {})
        # Ensure strategy_memory is a defaultdict
        strategy_memory = defaultdict(list)
        if 'strategy_memory' in loaded_data:
            strategy_memory.update(loaded_data['strategy_memory'])
        agent.strategy_memory = strategy_memory
        print("成功加载 Q 表和策略库")

    # 使用混合优化策略
    best_expr, best_reward, action_sequence = hybrid_optimize(agent, env, expression_id, similar_exprs)

    # 保存更新后的数据
    save_data = {
        'q_tables': agent.q_tables,
        'strategy_memory': dict(agent.strategy_memory)  # Convert to regular dict for saving
    }
    save_q_table(save_data)

    # 输出最终结果
    print("\nFinal results:")
    print(f"Original expression: {initial_expr}")
    print(f"Optimized expression: {best_expr}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Action sequence: {action_sequence}")
    return best_reward

if __name__ == "__main__":
    pass
    # initial_expr = "log(1 - x) / log(1 + x)"
    # size = 50000
    # light = 0.1
    # right = 0.9
    # oracle = get_expr_res(initial_expr, size, light, right, 113)
    #
    # env = ExpressionEnvironment(initial_expr)
    # agent = QLearningAgent(env.actions, learning_rate=0.2, discount_factor=0.95, epsilon=0.15)
    # expression_id = initial_expr
    #
    # # 尝试加载现有的 Q 表和策略库
    # loaded_data = load_q_table()
    # if loaded_data:
    #     agent.q_tables = loaded_data.get('q_tables', {})
    #     # Ensure strategy_memory is a defaultdict
    #     strategy_memory = defaultdict(list)
    #     if 'strategy_memory' in loaded_data:
    #         strategy_memory.update(loaded_data['strategy_memory'])
    #     agent.strategy_memory = strategy_memory
    #     print("成功加载 Q 表和策略库")
    #
    # # 使用混合优化策略
    # best_expr, best_reward, action_sequence = hybrid_optimize(agent, env, expression_id, initial_expr)
    #
    # # 保存更新后的数据
    # save_data = {
    #     'q_tables': agent.q_tables,
    #     'strategy_memory': dict(agent.strategy_memory)  # Convert to regular dict for saving
    # }
    # save_q_table(save_data)
    #
    # # 输出最终结果
    # print("\nFinal results:")
    # print(f"Original expression: {initial_expr}")
    # print(f"Optimized expression: {best_expr}")
    # print(f"Best reward achieved: {best_reward:.2f}")
    # print(f"Action sequence: {action_sequence}")