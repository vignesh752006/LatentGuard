import httpx
import pytest
import torch

from app.config.settings import Settings
from app.models.llama_planner import LlamaActionPlanner
from app.planning.representation import extract_spatial_features
from app.planning.pipeline import _decode_score_grid
from app.schemas.messages import ScoreGrid


@pytest.mark.asyncio
async def test_llama_prompt_includes_siglip_score_grid():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"action":"WAIT"}'}}]},
        )

    planner = LlamaActionPlanner(Settings(vdlm_llama_base_url="http://llama.test"))
    await planner._client.aclose()
    planner._client = httpx.AsyncClient(
        base_url="http://llama.test",
        transport=httpx.MockTransport(handler),
    )
    scores = [0.0] * 576
    scores[300] = 0.875
    try:
        await planner.plan_action(
            task_intent="find the account button",
            spatial_features=extract_spatial_features(torch.zeros(576, 1152), 24, 24, []),
            state_version=1,
            frame_id="f-1",
            last_action_id=None,
            score_grid=scores,
        )
    finally:
        await planner._client.aclose()

    assert "SigLIP task-to-patch scores computed from the sanitized tensor" in captured["body"]
    assert "0.8750" in captured["body"]


def test_score_grid_rejects_bad_base64_and_wrong_length():
    with pytest.raises(ValueError, match="valid base64"):
        _decode_score_grid(ScoreGrid(data="invalid!", shape=[24, 24]))
    with pytest.raises(ValueError, match="exactly 2304 bytes"):
        _decode_score_grid(ScoreGrid(data="AA==", shape=[24, 24]))


def test_score_grid_rejects_non_finite_scores():
    import base64
    import struct

    scores = [0.0] * 576
    scores[22] = float("nan")
    grid = ScoreGrid(
        data=base64.b64encode(struct.pack("<576f", *scores)).decode("ascii"),
        shape=[24, 24],
    )
    with pytest.raises(ValueError, match="non-finite"):
        _decode_score_grid(grid)
