from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.algorithms.classes.vpg import VPGConfig
from ray.rllib.examples.rl_modules.classes.vpg_torch_rlm import VPGTorchRLModule
from ray.tune import register_env
from ray.util.annotations import RayDeprecationWarning
from ray.rllib.core.columns import Columns
import torch
import warnings

warnings.filterwarnings("ignore", category=RayDeprecationWarning)

from pettingzoo.classic import chess_v6

register_env("chess_v6", lambda _: PettingZooEnv(chess_v6.env()))


class ChessRLModule(VPGTorchRLModule):

    def setup(self):
        # obs_space['observation'] is (8, 8, 111) for chess_v6
        obs_space = self.config.observation_space["observation"]
        act_space = self.config.action_space

        # Calculate input dimension for a simple flattened encoder
        input_dim = int(torch.prod(torch.tensor(obs_space.shape)))
        hidden_dim = self.model_config["hidden_dim"]
        output_dim = act_space.n

        self.policy_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def _forward(self, batch, **kwargs):
        obs_dict = batch[Columns.OBS]
        board_state = obs_dict["observation"].float()
        action_logits = self.policy_net(board_state)
        action_mask = obs_dict["action_mask"]
        # Apply action mask: set logits of invalid actions to -inf
        # This ensures they have zero probability after softmax
        action_logits = torch.where(
            action_mask.bool(),
            action_logits,
            torch.tensor(float('-inf'), dtype=action_logits.dtype, device=action_logits.device)
        )
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
            module_class=ChessRLModule,
        ),
    )  # custom config for the learner
    .environment("chess_v6")
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
