# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import copy
import logging
import traceback

import numpy as np
import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from . import image_tools

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        # Progress belongs to one client trajectory. Images and precomputed
        # latents are read-only and can be shared without copying model weights.
        session_tracker = copy.copy(getattr(self._policy, "subgoal_tracker", None))
        if session_tracker is not None:
            session_tracker.reset()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                previous_tracker = getattr(self._policy, "subgoal_tracker", None)
                if session_tracker is not None:
                    self._policy.subgoal_tracker = session_tracker
                try:
                    # The router is synchronous: no other handler can run while
                    # this connection's tracker is attached to the shared model.
                    ret = self._route_message(msg)
                finally:
                    if session_tracker is not None:
                        self._policy.subgoal_tracker = previous_tracker
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": "default",
                "error": {
                    "message": "Message must be a dict",
                    "message_type": type(msg).__name__,
                },
            }
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # reset
        elif mtype == "reset":
            try:
                if hasattr(self._policy, "reset_subgoal_tracker"):
                    self._policy.reset_subgoal_tracker()
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "reset_result",
                    "request_id": req_id,
                    "data": {"reset": True},
                }
            except Exception as e:
                logging.exception("Policy reset error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset_result",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }

        # infer
        elif mtype == "infer":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))}
                }
            try:
                if payload.get("state") is not None and not np.isfinite(np.asarray(payload["state"])).all():
                    raise ValueError("state must contain only finite values")
                # Decode both observation and optional goal images without replacing
                # the caller's arrays or modifying the request's nested containers.
                policy_payload = dict(payload)
                policy_payload["batch_images"] = image_tools.to_pil_preserve(payload["batch_images"])
                if payload.get("subgoal_images") is not None:
                    policy_payload["subgoal_images"] = image_tools.to_pil_preserve(payload["subgoal_images"])
                output_dict = self._policy.predict_action(**policy_payload)
                if "normalized_actions" in output_dict and not np.isfinite(
                    np.asarray(output_dict["normalized_actions"])
                ).all():
                    raise ValueError("Policy returned non-finite normalized_actions")
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)
                
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
