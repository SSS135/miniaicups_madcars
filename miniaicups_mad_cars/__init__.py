
try:
    from gym.envs.registration import register
    register(
        id='MadCarsAI-v0',
        entry_point='miniaicups_mad_cars.common.gym_env:MadCarsAIEnv',
        reward_threshold=0.95,
    )
except:
    pass