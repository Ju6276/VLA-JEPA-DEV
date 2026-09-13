"""Deployment overrides must take effect before constructing checkpoint models."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from deployment.model_server import server_policy


@pytest.mark.parametrize("override", [None, "/new/goals", ""])
@pytest.mark.parametrize("spatial_flags", [None, (True, False), (True, True)])
def test_goal_path_override_is_applied_before_checkpoint_construction(monkeypatch, override, spatial_flags):
    configured_path = "/old/goals"
    effective_path = configured_path if override is None else override
    calls = []

    class Policy:
        def __init__(self):
            if spatial_flags is not None:
                self.use_spatial_goal, self.use_spatial_memory = spatial_flags
            self.config = OmegaConf.create({"framework": {
                "delta_jepa": {"subgoals_path": effective_path},
                "action_model": {"state_dim": 2, "action_dim": 3, "action_horizon": 4},
            }})

        def to(self, *args):
            return self

        def eval(self):
            return self

        def load_subgoal_tracker(self, path):
            calls.append(("load_tracker", path))

    def load_checkpoint(path, **kwargs):
        # A stale path would fail here during construction, before main can
        # call load_subgoal_tracker. Verify the override reaches this boundary.
        observed_path = kwargs.get("config_overrides", {}).get(
            "framework.delta_jepa.subgoals_path", configured_path,
        )
        assert observed_path == effective_path
        calls.append(("checkpoint", path))
        return Policy()

    class Server:
        def __init__(self, **kwargs):
            calls.append(("server", kwargs["metadata"]))

        def serve_forever(self):
            calls.append(("serve",))

    monkeypatch.setattr(server_policy.baseframework, "from_pretrained", load_checkpoint)
    monkeypatch.setattr(server_policy, "WebsocketPolicyServer", Server)
    monkeypatch.setattr(server_policy.socket, "gethostbyname", lambda name: "127.0.0.1")
    server_policy.main(SimpleNamespace(
        ckpt_path="local.pt", cuda=0, use_bf16=False, subgoals_path=override,
        use_verifier=False, port=10093,
    ))

    assert calls[0] == ("checkpoint", "local.pt")
    assert calls[-1] == ("serve",)
    assert [call[1] for call in calls if call[0] == "load_tracker"] == (
        [effective_path] if effective_path else []
    )
    metadata = next(call[1] for call in calls if call[0] == "server")
    expected_flags = spatial_flags or (False, False)
    assert metadata["spatial_goal_enabled"] is expected_flags[0]
    assert metadata["spatial_memory_enabled"] is expected_flags[1]
