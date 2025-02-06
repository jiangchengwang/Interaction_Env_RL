import os
import sys
from gymnasium.envs.registration import register


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    register(
        id="interaction-v0",
        entry_point="simulation_environment.osm_envs.interaction_env:InterActionEnv",
    )

    register(
        id="interaction-v1",
        entry_point="simulation_environment.osm_envs.interaction_env:InterActionV1Env",
    )

_register_highway_envs()