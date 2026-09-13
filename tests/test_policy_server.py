"""Exercise the policy router with the same NumPy wire format as deployment."""

import numpy as np
import pytest
from PIL import Image

from deployment.model_server.tools import msgpack_numpy
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
