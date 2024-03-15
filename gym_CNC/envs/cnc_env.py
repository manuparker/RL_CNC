import gymnasium as gym
import numpy as np
from typing import Optional, Any
from gymnasium.error import DependencyNotInstalled
from gymnasium import spaces

from gym_CNC.envs.tool_path import ToolPath

""" 这个是继承于gym的强化学习自定义环境的编写，在这个环境中，需要创造一个tool_path对象 """
""" 在这个类中，只负责实现reset、step、render等在强化学习环境中的基本操作，他们均通过调用tool_path中的方法来实现 """
""" 具体底层的该CNC环境中的计算各个量的算法，均在tool_path类中实现 """


class CNCEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, tool_path=ToolPath()):
        self.tool_path = tool_path
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 800
        self.isopen = True

        self.low_state = np.array(
            [-np.pi/2, 0, -np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        self.high_state = np.array(
            [np.pi/2, 20, np.pi/2, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
        )

        self.action_space = spaces.Box(
            low=np.array([-np.pi/2, 0]),
            high=np.array([np.pi/2, 20]),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        obs = self.tool_path.init_state()

        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        obs, reward = self.tool_path.action(action)
        terminated, truncated = self.tool_path.is_done()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.tool_path.render(self)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
