from collections import defaultdict
from common.gridworld import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):        # 给策略、分数表、地图、打折率
    for state in env.states():                  # 12 个格子挨个走一遍（书里包含墙）
        if state == env.goal_state:             # 苹果格：到了就结束
            V[state] = 0                        # 所以分数固定 0
            continue                            # 跳过，看下一格
        action_probs = pi[state]                # 这格四个方向各 0.25
        new_V = 0                               # 准备累加
        for action, action_prob in action_probs.items():   # 四个方向挨个算
            next_state = env.next_state(state, action)     # 问地图：会到哪格
            r = env.reward(state, action, next_state)      # 问地图：这步给几分
            new_V += action_prob * (r + gamma * V[next_state])  # 概率 ×(当场分 + 打九折的下一格分)
        V[state] = new_V                        # 累加结果写回这格
    return V


def policy_eval(pi, V, env, gamma, threshold=0.000000000000000000000001):
    while True:
        old_V = V.copy()                       # 先把上一轮分数抄一份
        V = eval_onestep(pi, V, env, gamma)    # 算一轮
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])  # 每个格子改了多少
            if delta < t:
                delta = t                      # 记下最大改动
        if delta < threshold:                  # 改动小于 0.001 就收工
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # 策略：四个方向各 1/4
    V = defaultdict(lambda: 0)                                        # 分数表：没见过的格子自动给 0

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)                        # 画出书里图 4-13 那样的热力图 + 箭头
