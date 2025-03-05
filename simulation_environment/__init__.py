import os
import sys
from gymnasium.envs.registration import register


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    register(
        id="interaction-v0",
        entry_point="simulation_environment.osm_envs.interaction_env:InteractionEnv",
    )

    register(
        id="interaction-v1",
        entry_point="simulation_environment.osm_envs.interaction_env:InteractionV1Env",
    )

    register(
        id="interaction-eval-v0",
        entry_point="simulation_environment.osm_envs.interaction_env:InteractionEvalEnv",
    )

    register(
        id="interaction-rl-v0",
        entry_point="simulation_environment.osm_envs.interaction_gail_env:InteractionGAILEnv",
    )


_register_highway_envs()