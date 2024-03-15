from gymnasium.envs.registration import register

register(
    id='CNC-v0',
    entry_point='gym_CNC.envs:CNCEnv',
    max_episode_steps=2000,
)