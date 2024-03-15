import collections
import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def moving_average(a, window_size):
    """使用moving_average的方法对数据进行平滑处理，a为传入的return数据列表，window_size为取平均的个数"""
    """以window_size=9为例，该平均算法的逻辑是，每个位置，取自己和前面4个数据以及后面4个数据共9个数据进行平均操作"""
    """针对前4个数据与后4个数据，该算法做特殊处理"""
    """第一个数据：直接用其本身"""
    """第二个数据：用前3个数据之和除以3"""
    """第三个数据：用前5个数据之和除以5"""
    """第四个数据：用前7个数据之和除以7"""
    """倒数第一个数据：直接用其本身"""
    """倒数第二个数据：用后3个数据之和除以3"""
    """倒数第三个数据：用后5个数据之和除以5"""
    """第四个数据：用后7个数据之和除以7"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  # 在a前插入0之后，算每个位置之前的所有元素的和
    # 加入0的作用是能够得到第1个元素到第9个元素的累积和
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size  # 利用累计和切片：（第9个元素到最后 - 第1个元素到倒数第10个元素）/ 9 = 第5个位置到倒数第5个位置的均值
    r = np.arange(1, window_size - 1, 2)  # 得到r = [1 ,3, 5, 7]
    begin = (
        np.cumsum(a[: window_size - 1])[::2] / r
    )  # 得到前8个元素的和并分别取1、3、5、7位置的和除以r作为前4个位置的均值
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[
        ::-1
    ]  # 得到后8个元素的和并分别取最后1、3、5、7位置的和除以r作为后4个位置的均值
    if window_size % 2 == 0:  # 该条件补充了window_size为偶数的情况，若不加该语句，则偶数时平滑后的输出比原数组元素少1
        end = np.insert(
            end,
            -int((window_size / 2)),
            (cumulative_sum[-1] - cumulative_sum[-window_size]) / (r[-1] + 2),
        )
    return np.concatenate((begin, middle, end))  # 将其拼到一起，则为平滑后的数据


def compute_advantage(gamma, lmbda, td_delta):
    """用来计算广义优势估计"""
    # td_delta为每一个时间步的时序差分误差
    # 它等于当前时刻的reward加上gamma乘下一个状态的state value，再减去这个时间步的state value
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_np = np.array(advantage_list)
    return torch.tensor(advantage_np, dtype=torch.float)


def train_off_policy_agent(
    env, agent, num_episodes, replay_buffer, minimal_size, batch_size
):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                state = state[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    env.render()
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, noise_scale=None):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        self.noise_scale = noise_scale if noise_scale is not None else [bound / 10.0 for bound in action_bound]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions = torch.tanh(self.fc2(x))

        action1 = actions[:, 0] * self.action_bound[0]  # -π/2到π/2
        action2 = (actions[:, 1] + 1) / 2 * self.action_bound[1]  # 0到20

        # 使用 torch.stack 或 torch.cat 来组合动作，避免原地修改
        adjusted_actions = torch.stack([action1, action2], dim=1)

        return adjusted_actions


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], 1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    """DDPG算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        sigma,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        device,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(
            device
        )
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差，均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.state_dim = state_dim

    def take_action(self, state):
        state = (
            torch.tensor(state, dtype=torch.float).view(-1, len(state)).to(self.device)
        )
        action = self.actor(state).detach().cpu().numpy()  # 获取动作
        # 根据每个动作的噪声标准差添加噪声
        noise = np.array(
            [np.random.normal(0, scale) for scale in self.actor.noise_scale]
        )
        action_with_noise = action + noise
        action = action_with_noise.squeeze()

        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def update(self, transition_dict):
        states_np = np.array(transition_dict["states"])
        next_states_np = np.array(transition_dict["next_states"])
        actions_np = np.array(transition_dict["actions"])
        rewards_np = np.array(transition_dict["rewards"])

        states = torch.tensor(states_np, dtype=torch.float).to(self.device)
        actions = (
            torch.tensor(actions_np, dtype=torch.float).view(-1, 2).to(self.device)
        )
        rewards = (
            torch.tensor(rewards_np, dtype=torch.float).view(-1, 1).to(self.device)
        )
        next_states = torch.tensor(next_states_np, dtype=torch.float).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
