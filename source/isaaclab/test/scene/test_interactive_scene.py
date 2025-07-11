# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # articulation
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/revolute_articulation.usd"),
        actuators={
            "joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0),
        },
    )
    # rigid object
    rigid_obj = RigidObjectCfg(
        prim_path="/World/RigidObj",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )

    # sensor
    sensor = ContactSensorCfg(
        prim_path="/World/Robot",
    )
    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(),
    )


@pytest.fixture(scope="module")
def setup_scene():
    """Fixture to set up scene parameters."""
    sim_dt = 0.001
    scene_cfg = MySceneCfg(num_envs=1, env_spacing=1)
    return sim_dt, scene_cfg


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_scene_entity_isolation(device, setup_scene):
    """Tests that multiple instances of InteractiveScene do not share any data.

    In this test, two InteractiveScene instances are created in a loop and added to a list.
    The scene at index 0 of the list will have all of its entities cleared manually, and
    the test compares that the data held in the scene at index 1 remained intact.
    """
    sim_dt, scene_cfg = setup_scene
    scene_list = []
    # create two InteractiveScene instances
    for _ in range(2):
        with build_simulation_context(device=device, dt=sim_dt) as _:
            scene = InteractiveScene(scene_cfg)
            scene_list.append(scene)
    scene_0 = scene_list[0]
    scene_1 = scene_list[1]
    # clear entities for scene_0 - this should not affect any data in scene_1
    scene_0.articulations.clear()
    scene_0.rigid_objects.clear()
    scene_0.sensors.clear()
    scene_0.extras.clear()
    # check that scene_0 and scene_1 do not share entity data via dictionary comparison
    assert scene_0.articulations == dict()
    assert scene_0.articulations != scene_1.articulations
    assert scene_0.rigid_objects == dict()
    assert scene_0.rigid_objects != scene_1.rigid_objects
    assert scene_0.sensors == dict()
    assert scene_0.sensors != scene_1.sensors
    assert scene_0.extras == dict()
    assert scene_0.extras != scene_1.extras
