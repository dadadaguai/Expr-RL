import random
import numpy as np
from sympy import (sympify, expand, factor, cancel, simplify,
                   expand_trig, trigsimp, expand_log, logcombine,
                   expand_power_base, powsimp, powdenest, together,
                   fraction, symbols, apart, log, preorder_traversal, Function, exp)
import re
import pickle

from 强化学习.Tools import get_expr_res, evaluate_expression_original
from 强化学习.表达式重写生成Tools import custom_distribute, optimize_log1p, log_exp_replace, optimize_expm1, \
    sqrt_rewrite, smart_rewrite, associative_law
from Tools import cache_evaluate

# 定义符号
x = symbols('x')

ulp_cache = {}

# 精度优化函数
def evaluate_precision(current_expr, new_expr):
    """
    评估表达式的精度优化效果。
    返回一个分数，表示新表达式相对于旧表达式的精度优化程度。
    """
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
    def __init__(self, initial_expr):
        self.initial_expr = initial_expr
        self.current_expr = sympify(initial_expr)
        self.actions = self.determine_actions(initial_expr)
        self.state_manager = StateManager()
        self.state_manager.reset(str(self.current_expr))

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
        best_expr = max(new_exprs, key=lambda expr: evaluate_precision(old_expr, expr))
        changed = (old_expr != best_expr)
        if not changed:
            reward = -1
        else:
            optimization_value = evaluate_precision(old_expr, best_expr)
            reward = optimization_value + self.state_manager.current_optimization if optimization_value > 0 else optimization_value
        if changed and reward >= 0:
            self.current_expr = best_expr
            self.state_manager.update(str(self.current_expr), action.__name__, optimization_value)
        return str(self.current_expr), reward, changed


class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.01):
        self.actions = actions
        self.q_tables = {}  # 分层 Q 表
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.last_action = None

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


def train_agent(env, agent, expression_id, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}:")
        while not done and steps < 20:
            action = agent.choose_action(expression_id, state)
            next_state, reward, changed = env.step(action)

            agent.update_q_value(expression_id, state, action, reward, next_state)

            episode_reward += reward
            state = next_state
            steps += 1

            print(f"Step {steps}: Action: {action.__name__}, Reward: {reward:.2f}")

            if reward < 0 or not changed:
                done = True

        print(f"Episode {episode + 1} completed. Total reward: {episode_reward:.2f}")


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


def save_q_table(q_tables, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(q_tables, f)
    print(f"Q表已保存到 {filename}")


def load_q_table(filename="q_table.pkl"):
    try:
        with open(filename, "rb") as f:
            q_tables = pickle.load(f)
        print(f"Q表已从 {filename} 加载")
        return q_tables
    except FileNotFoundError:
        print(f"未找到文件 {filename}，将使用空Q表")
        return {}


def optimize_expression(agent, env, expression_id, expression):
    state = env.reset(expression)
    best_expr = state
    best_reward = float('-inf')
    done = False
    action_sequence = []
    steps = 0

    print(f"\nOptimizing Expression: {expression}")

    while not done and steps < 15:
        action = agent.choose_action(expression_id, state)
        next_state, reward, changed = env.step(action)

        print(f"Step {steps + 1}: Action: {action.__name__}, Reward: {reward:.2f}, State: {next_state}")

        if reward > best_reward:
            best_expr = next_state
            best_reward = reward

        if reward >= 0 and changed:
            action_sequence.append(action.__name__)
        else:
            print("Negative reward or no change detected. Considering stopping.")

        state = next_state
        steps += 1

        if reward < 0 and len(action_sequence) >= 3:
            done = True

    print(f"\nOptimization complete. Best reward: {best_reward:.2f}")
    print(f"Optimized expression: {best_expr}")
    print(f"Action sequence: {action_sequence}")
    return best_expr, best_reward, action_sequence


if __name__ == "__main__":
    initial_expr = "log(1-x)/log(1+x)"
    size = 50000
    light = 0.01
    right = 0.5
    oracle = get_expr_res(initial_expr, size, light, right, 113)

    env = ExpressionEnvironment(initial_expr)
    agent = QLearningAgent(env.actions, learning_rate=0.2, discount_factor=0.95, epsilon=0.15)
    expression_id = initial_expr

    # 加载现有的 Q 表
    loaded_q_tables = load_q_table()
    if loaded_q_tables:
        agent.q_tables = loaded_q_tables

    # 检查是否已经存在该表达式的 Q 表
    if expression_id in agent.q_tables:
        print(f"表达式 {expression_id} 已存在，直接进行验证。")
    else:
        print(f"表达式 {expression_id} 不存在，开始训练。")
        # 训练智能体
        train_agent(env, agent, expression_id, episodes=500)
        # 保存训练后的 Q 表
        save_q_table(agent.q_tables)

    # 优化表达式
    print("\nStarting optimization...")
    best_expr, best_reward, action_sequence = optimize_expression(agent, env, expression_id, initial_expr)

    print("\nFinal results:")
    print(f"Original expression: {initial_expr}")
    print(f"Optimized expression: {best_expr}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Action sequence: {action_sequence}")