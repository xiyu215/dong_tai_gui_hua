# 书页 103～106：4.4 实施策略迭代法  ch04/policy_iter.py
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval


def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]  # ①
            action_values[action] = value

        max_action = argmax(action_values)  # ②
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs  # ③
    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)  # ①评估
        new_pi = greedy_policy(V, env, gamma)  # ②改进

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:  # ③更新检查
            break
        pi = new_pi

    return pi


if __name__ == '__main__':
    action_values = {0: 0.1, 1: -0.3, 2: 9.9, 3: -1.3}  # 9.9 是最大值（键是 2）
    max_action = argmax(action_values)
    print(max_action)

    env = GridWorld()
    gamma = 0.9
    # 书页 106 原为 policy_iter(env, gamma)；这里显式打开渲染以复现图 4-16
    pi = policy_iter(env, gamma, is_render=True)
