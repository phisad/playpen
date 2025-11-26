from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.algorithms.classes.vpg import VPGConfig
from ray.rllib.examples.rl_modules.classes.vpg_torch_rlm import VPGTorchRLModule
from ray.tune import register_env
from ray.util.annotations import RayDeprecationWarning
from ray.rllib.core.columns import Columns
import torch
import warnings

from playpen.adapters import pettingzoo

warnings.filterwarnings("ignore", category=RayDeprecationWarning)

register_env("taboo", lambda _: PettingZooEnv(pettingzoo.env("taboo", single_pass=False)))


class TabooRLModule(VPGTorchRLModule):

    def setup(self):
        # obs_space['content'] length is 8192
        # obs_space['role'] length is 128
        obs_space = self.config.observation_space
        act_space = self.config.action_space

        # Calculate input dimension for a simple flattened encoder
        input_dim = int(obs_space["role"].max_length + obs_space["content"].max_length)
        hidden_dim = self.model_config["hidden_dim"]
        output_dim = act_space.max_length

        self.policy_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def _forward(self, batch, **kwargs):
        obs_dict = batch[Columns.OBS]
        role = obs_dict["role"]  # ignore action-mask
        content = obs_dict["content"]  # ignore action-mask
        print(role, content)
        # todo: needs to be tokenized here
        action_logits = self.policy_net(role + content)
        return {
            Columns.ACTION_DIST_INPUTS: action_logits
        }

    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)


config = (
    VPGConfig()
    .framework("torch")
    .multi_agent(
        policies={"shared_policy"},  # Both players use the same policy
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
    )
    .rl_module(
        model_config={"hidden_dim": 64},
        rl_module_spec=RLModuleSpec(
            module_class=TabooRLModule,
        ),
    )  # custom config for the learner
    .environment("taboo")
    .env_runners(
        num_env_runners=0,
    )
    .training(
        num_episodes_per_train_batch=2,
        num_epochs=1
    )
)

print("Building algo")
algo = config.build_algo()
print("Starting training")
algo.train()
print("Evaluating")
algo.evaluate()
print("Stopping")
algo.stop()
