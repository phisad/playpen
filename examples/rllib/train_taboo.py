import torch
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.algorithms.classes.vpg import VPGConfig
from ray.rllib.examples.rl_modules.classes.vpg_torch_rlm import VPGTorchRLModule
from ray.tune import register_env
from ray.util.annotations import RayDeprecationWarning
from ray.rllib.core.columns import Columns
import warnings

from playpen.adapters import pettingzoo

warnings.filterwarnings("ignore", category=RayDeprecationWarning)

register_env("taboo", lambda _: PettingZooEnv(pettingzoo.env("taboo", single_pass=False)))

# todo: set the vocab_size in the env based on the loaded model (which is only loaded during module setup)

class TabooRLModule(VPGTorchRLModule):

    def setup(self):
        # Initialize a huggingface model
        # Note: We cannot use clem/backends because that only considers inference use cases
        self.model = None

        act_space = self.config.action_space
        self.model.set_gen_args(max_tokens=act_space.max_length, temperature=0.7)

        # obs_space['content'] length is 8192
        # obs_space['role'] length is 128
        obs_space = self.config.observation_space
        input_dim = int(obs_space["role"].max_length + obs_space["content"].max_length)

        # check that the huggingface model supports input_dim
        assert input_dim <= self.model.config.max_position_embeddings

    def _forward(self, batch, **kwargs):
        obs_dict = batch[Columns.OBS]
        _, response, _ = self.model.generate_response([obs_dict])  # obs_dict has role and content keys
        token_logits = response["logits"]  # token logits over the vocabulary at each token position
        log_probs = torch.log_softmax(token_logits, dim=-1)

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
