import pytest


def test_base_reward(base_reward):
    with pytest.raises(NotImplementedError):
        base_reward()


@pytest.mark.parametrize('reward_name',
                         [('linear_reward'),
                          ('exponential_reward'),
                          ('hourly_linear_reward'),
                          ])
def test_rewards(reward_name, env_demo_continuous, request):
    reward = request.getfixturevalue(reward_name)
    env_demo_continuous.reset()
    a = env_demo_continuous.action_space.sample()
    obs, _, _, _, _ = env_demo_continuous.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    obs_dict = dict(zip(env_demo_continuous.variables['observation'], obs))
    R, terms = reward(obs_dict)
    assert R <= 0
    assert env_demo_continuous.reward_fn.W_energy == 0.5
    assert isinstance(terms, dict)
    assert len(terms) > 0


def test_custom_reward(custom_reward):
    R, terms = custom_reward()
    assert R == -1.0
    assert isinstance(terms, dict)
    assert len(terms) == 0
