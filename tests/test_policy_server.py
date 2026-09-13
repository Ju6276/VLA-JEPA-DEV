"""Exercise the policy router with the same NumPy wire format as deployment."""

import asyncio
import numpy as np
import pytest
from PIL import Image
from websockets.exceptions import ConnectionClosedOK

from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class ImagePolicy:
    def __init__(self):
        self.payload = None

    def predict_action(self, **payload):
        for views in payload["batch_images"]:
            for image in views:
                assert isinstance(image, Image.Image)
        if payload.get("subgoal_images") is not None:
            for image in payload["subgoal_images"]:
                assert isinstance(image, Image.Image)
        self.payload = payload
        return {"actions": np.zeros((1, 2, 3), dtype=np.float32)}


def wire_roundtrip(value):
    return msgpack_numpy.unpackb(msgpack_numpy.packb(value))


def test_numpy_goal_request_through_wire_and_router():
    image = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    goal = image[::-1].copy()
    request = wire_roundtrip({
        "type": "infer",
        "request_id": "explicit-goal",
        "payload": {
            "batch_images": [[image]],
            "subgoal_images": [goal],
            "instructions": ["Put the cup on the tray"],
        },
    })
    policy = ImagePolicy()
    response = wire_roundtrip(WebsocketPolicyServer(policy)._route_message(request))

    assert response["status"] == "ok"
    assert response["ok"] is True
    assert response["type"] == "inference_result"
    assert response["request_id"] == "explicit-goal"
    np.testing.assert_array_equal(response["data"]["actions"], np.zeros((1, 2, 3)))
    np.testing.assert_array_equal(np.asarray(policy.payload["batch_images"][0][0]), image)
    np.testing.assert_array_equal(np.asarray(policy.payload["subgoal_images"][0]), goal)
    assert isinstance(request["payload"]["batch_images"][0][0], np.ndarray)
    assert isinstance(request["payload"]["subgoal_images"][0], np.ndarray)


@pytest.mark.parametrize("goal_fields", [{}, {"subgoal_images": None}])
def test_optional_goal_keeps_automatic_goal_requests_compatible(goal_fields):
    policy = ImagePolicy()
    payload = {"batch_images": [[np.zeros((4, 5, 3), dtype=np.uint8)]], **goal_fields}
    request = wire_roundtrip(payload)
    response = WebsocketPolicyServer(policy)._route_message(request)

    assert response["ok"] is True
    assert ("subgoal_images" in policy.payload) == ("subgoal_images" in goal_fields)
    assert policy.payload.get("subgoal_images") is None


def test_direct_router_preserves_pil_images_and_nested_container_layout():
    current = Image.new("RGB", (4, 5))
    goal = Image.new("RGB", (7, 6))
    payload = {"batch_images": ([current],), "subgoal_images": (goal,)}
    policy = ImagePolicy()
    response = WebsocketPolicyServer(policy)._route_message({"payload": payload})

    assert response["ok"] is True
    assert isinstance(policy.payload["batch_images"], tuple)
    assert isinstance(policy.payload["batch_images"][0], list)
    assert isinstance(policy.payload["subgoal_images"], tuple)
    assert policy.payload["batch_images"][0][0] is current
    assert policy.payload["subgoal_images"][0] is goal
    assert policy.payload["batch_images"] is not payload["batch_images"]
    assert policy.payload["batch_images"][0] is not payload["batch_images"][0]


@pytest.mark.parametrize("bad_field", ["batch_images", "subgoal_images"])
def test_bad_images_keep_protocol_error_response_and_do_not_mutate_request(bad_field):
    current = np.zeros((4, 5, 3), dtype=np.uint8)
    invalid = np.zeros((4, 5), dtype=np.uint8)
    payload = {"batch_images": [[current]], "subgoal_images": [current]}
    payload[bad_field] = [[invalid]] if bad_field == "batch_images" else [invalid]
    policy = ImagePolicy()
    response = wire_roundtrip(WebsocketPolicyServer(policy)._route_message({
        "type": "infer", "request_id": "invalid-image", "payload": payload,
    }))

    assert response["ok"] is False
    assert response["status"] == "error"
    assert response["type"] == "inference_result"
    assert response["request_id"] == "invalid-image"
    assert "Expected 3D array" in response["error"]["message"]
    assert isinstance(payload["batch_images"][0][0], np.ndarray)
    assert isinstance(payload["subgoal_images"][0], np.ndarray)
    assert policy.payload is None


@pytest.mark.parametrize("message", [None, [], [1], "infer", 42, True])
def test_non_dictionary_message_returns_protocol_error(message):
    server = WebsocketPolicyServer(ImagePolicy())
    response = wire_roundtrip(server._route_message(wire_roundtrip(message)))

    assert response["ok"] is False
    assert response["status"] == "error"
    assert response["error"]["message"] == "Message must be a dict"
    assert server._route_message({"type": "ping"})["ok"] is True


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.5])
def test_state_must_be_finite_before_calling_policy(value):
    policy = ImagePolicy()
    response = WebsocketPolicyServer(policy)._route_message(wire_roundtrip({
        "type": "infer", "request_id": "state-check", "payload": {
            "batch_images": [[np.zeros((4, 5, 3), dtype=np.uint8)]],
            "state": np.asarray([[[value, 0.0]]], dtype=np.float32),
        },
    }))

    assert response["request_id"] == "state-check"
    if np.isfinite(value):
        assert response["ok"] is True
        assert policy.payload is not None
    else:
        assert response["ok"] is False
        assert "state must contain only finite values" in response["error"]["message"]
        assert policy.payload is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.5])
def test_nonfinite_policy_actions_are_not_reported_as_success(value):
    class ActionPolicy:
        def predict_action(self, **payload):
            return {"normalized_actions": np.full((1, 2, 3), value, dtype=np.float32)}

    response = wire_roundtrip(WebsocketPolicyServer(ActionPolicy())._route_message({
        "type": "infer", "request_id": "output-check", "payload": {
            "batch_images": [[np.zeros((4, 5, 3), dtype=np.uint8)]],
        },
    }))

    assert response["request_id"] == "output-check"
    if np.isfinite(value):
        assert response["ok"] is True
        assert np.isfinite(response["data"]["normalized_actions"]).all()
    else:
        assert response["ok"] is False
        assert "non-finite normalized_actions" in response["error"]["message"]
        assert "data" not in response


class RouterSocket:
    """Carry the real client's serialized requests through the real router."""

    def __init__(self, server):
        self.server = server

    def send(self, frame):
        self.response = self.server._route_message(msgpack_numpy.unpackb(frame))

    def recv(self):
        return msgpack_numpy.packb(self.response)


@pytest.mark.parametrize("fail", [False, True])
def test_client_reset_reaches_policy_and_surfaces_failures(fail):
    class ResetPolicy:
        reset_count = 0

        def reset_subgoal_tracker(self):
            if fail:
                raise ValueError("cannot reset tracker")
            self.reset_count += 1

    policy = ResetPolicy()
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()
    client._ws = RouterSocket(WebsocketPolicyServer(policy))

    if fail:
        with pytest.raises(RuntimeError, match="cannot reset tracker"):
            client.reset("new instruction")
        assert policy.reset_count == 0
    else:
        response = client.reset("new instruction")
        assert response["type"] == "reset_result"
        assert response["data"]["reset"] is True
        assert policy.reset_count == 1


def test_websocket_handlers_isolate_tracker_progress_and_recover_from_bad_requests():
    import torch
    from starVLA.tools.subgoal_tracker import SubgoalTracker

    original_tracker = SubgoalTracker(
        [Image.new("RGB", (2, 2)) for _ in range(3)], z_goals=torch.eye(3),
    )

    class TrackedPolicy:
        subgoal_tracker = original_tracker

        def predict_action(self, **payload):
            # Shallow session copies share the read-only visual resources.
            assert self.subgoal_tracker.subgoal_images is original_tracker.subgoal_images
            assert self.subgoal_tracker.z_goals is original_tracker.z_goals
            if payload.get("fail"):
                raise ValueError("request failed")
            self.subgoal_tracker.update(torch.tensor(payload["latent"].copy()))
            return self.subgoal_tracker.state_dict()

        def reset_subgoal_tracker(self):
            self.subgoal_tracker.reset()

    class Connection:
        """In-memory transport drives the actual async handler without ports."""

        remote_address = ("test", 0)

        def __init__(self):
            self.incoming = asyncio.Queue()
            self.outgoing = asyncio.Queue()

        async def recv(self):
            frame = await self.incoming.get()
            if frame is None:
                raise ConnectionClosedOK(None, None)
            return frame

        async def send(self, frame):
            await self.outgoing.put(frame)

        async def close(self, **kwargs):
            pass

    async def exercise():
        policy = TrackedPolicy()
        server = WebsocketPolicyServer(policy)
        a, b = Connection(), Connection()
        handlers = [asyncio.create_task(server._handler(connection)) for connection in (a, b)]

        async def request(connection, message):
            await connection.incoming.put(msgpack_numpy.packb(message))
            frame = await asyncio.wait_for(connection.outgoing.get(), timeout=2)
            assert policy.subgoal_tracker is original_tracker
            assert original_tracker.current_index == 0
            return msgpack_numpy.unpackb(frame)

        def infer(latent=(0.0, 0.0, 1.0), **extra):
            return {"type": "infer", "payload": {
                "batch_images": [[np.zeros((2, 2, 3), dtype=np.uint8)]],
                "latent": np.asarray(latent, dtype=np.float32), **extra,
            }}

        try:
            for connection in (a, b):
                await asyncio.wait_for(connection.outgoing.get(), timeout=2)  # metadata
            assert (await request(a, infer((1.0, 0.0, 0.0))))["data"]["current_index"] == 1
            assert (await request(b, infer()))["data"]["current_index"] == 0
            assert (await request(b, {"type": "reset"}))["ok"] is True
            assert (await request(a, infer()))["data"]["current_index"] == 1
            assert (await request(a, infer(fail=True)))["ok"] is False
            assert (await request(a, []))["ok"] is False
            assert (await request(a, {"type": "ping"}))["ok"] is True
            assert (await request(a, infer()))["data"]["current_index"] == 1
        finally:
            for connection in (a, b):
                await connection.incoming.put(None)
            await asyncio.gather(*handlers)

    asyncio.run(exercise())
