import unittest

from playpen.adapters import pettingzoo


class PettingZooTestCase(unittest.TestCase):

    def test_loop(self):
        mode = "auto"

        idx_to_agent = {
            "guesser": "mock-guesser",
            "describer": "mock-describer"
        }  # <- gamer_master.get_player_mapping()

        env = pettingzoo.env("taboo")
        env.reset()  # use default instance
        for agent_id in env.agent_iter(max_iter=10):
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            elif mode == "mock":
                action = env.action_space(agent_id).sample()
            elif mode == "auto":
                action = agent_id(observation)
            else:
                action = idx_to_agent[agent_id](observation)
            print(agent_id.name, action)
            env.step(action)
        env.close()


if __name__ == '__main__':
    unittest.main()
