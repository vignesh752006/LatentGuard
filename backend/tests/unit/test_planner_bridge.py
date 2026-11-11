"""Task 6 — Planner bridge unit tests.

Tests:
- JSON repair strategies
- MockActionPlanner with score_grid
- Pipeline retry logic via fake LLM server (FakePlanner)
"""

from __future__ import annotations

import pytest

from app.planning.json_repair import PlannerResponseError, repair_json
from app.planning.representation import SpatialReasoningFeatures
from app.models.mock_planner import MockActionPlanner


# ---- JSON repair tests --------------------------------------------------------

def test_repair_valid_json():
    obj = repair_json('{"action": "CLICK", "confidence": 0.9}')
    assert obj["action"] == "CLICK"


def test_repair_markdown_fenced():
    raw = "```json\n{\"action\": \"SCROLL\", \"confidence\": 0.7}\n```"
    obj = repair_json(raw)
    assert obj["action"] == "SCROLL"


def test_repair_markdown_no_lang():
    raw = "```\n{\"action\": \"DONE\"}\n```"
    obj = repair_json(raw)
    assert obj["action"] == "DONE"


def test_repair_prefix_garbage():
    raw = "Sure! Here is the JSON: {\"action\": \"NAVIGATE\", \"text\": \"https://x.com\"}"
    obj = repair_json(raw)
    assert obj["action"] == "NAVIGATE"


def test_repair_trailing_comma():
    raw = '{"action": "CLICK", "confidence": 0.9,}'
    obj = repair_json(raw)
    assert obj["action"] == "CLICK"


def test_repair_empty_raises():
    with pytest.raises(PlannerResponseError):
        repair_json("")


def test_repair_no_braces_raises():
    with pytest.raises(PlannerResponseError):
        repair_json("I cannot decide on an action.")


# ---- MockActionPlanner with score_grid ----------------------------------------

def _make_features(n_patches: int = 576) -> SpatialReasoningFeatures:
    return SpatialReasoningFeatures(
        patch_count=n_patches,
        grid_rows=24,
        grid_cols=24,
        patch_norms=[1.0] * min(24, n_patches),
        row_activation_means=[0.5] * 24,
        col_activation_means=[0.5] * 24,
        dominant_patch_index=100,
        dominant_patch_coords=(0.5, 0.5),
        sensitive_patch_count=10,
        tensor_mean=0.0,
        tensor_std=1.0,
    )


@pytest.mark.asyncio
async def test_mock_planner_uses_score_grid_for_click():
    planner = MockActionPlanner()
    features = _make_features()
    # Score grid with highest score at patch 200
    score_grid = [0.0] * 576
    score_grid[200] = 1.0
    result = await planner.plan_action(
        task_intent="click something",
        spatial_features=features,
        state_version=1,
        frame_id="f1",
        last_action_id=None,
        score_grid=score_grid,
    )
    assert result["action"] == "CLICK"
    assert result["target_id"] == 200


@pytest.mark.asyncio
async def test_mock_planner_fallback_no_score_grid():
    planner = MockActionPlanner()
    features = _make_features()
    result = await planner.plan_action(
        task_intent="click something",
        spatial_features=features,
        state_version=1,
        frame_id="f2",
        last_action_id=None,
        score_grid=None,
    )
    assert result["action"] == "CLICK"
    assert result["target_id"] == features.dominant_patch_index


@pytest.mark.asyncio
async def test_mock_planner_scroll_intent():
    planner = MockActionPlanner()
    features = _make_features()
    result = await planner.plan_action(
        task_intent="scroll down the page",
        spatial_features=features,
        state_version=1,
        frame_id="f3",
        last_action_id=None,
    )
    assert result["action"] == "SCROLL"


@pytest.mark.asyncio
async def test_mock_planner_navigate_intent():
    planner = MockActionPlanner()
    features = _make_features()
    result = await planner.plan_action(
        task_intent="navigate to http://example.com",
        spatial_features=features,
        state_version=1,
        frame_id="f4",
        last_action_id=None,
    )
    assert result["action"] == "NAVIGATE"


# ---- Pipeline retry via fake LLM -----------------------------------------------

from app.planning.pipeline import ActionPlanningPipeline


class FakePlanner:
    """Returns a raw string on first call, valid dict on retry."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def plan_action(self, **_kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    async def health_check(self):
        return True


from app.schemas.messages import (
    ActionContext,
    EncoderSpec,
    LatentFramePayload,
    PatchGrid,
    SanitizationProvenance,
    SpatialMetadata,
    TensorPayload,
    Viewport,
)
from app.session.browser_state import BrowserSession
import torch


def _make_frame() -> LatentFramePayload:
    tensor_payload = TensorPayload(
        encoding="base64",
        dtype="float16",
        shape=[576, 1152],
        data="AA==",
        compression="none",
    )
    return LatentFramePayload(
        frame_id="test-frame-1",
        state_version=1,
        tensor=tensor_payload,
        spatial_metadata=SpatialMetadata(
            patch_grid=PatchGrid(rows=24, cols=24),
            sensitive_patch_indices=[0, 1],
            viewport_hash=None,
        ),
        action_context=ActionContext(),
        provenance=SanitizationProvenance(),
    )


def _make_session() -> BrowserSession:
    import time
    return BrowserSession(
        session_id="sess-1",
        task_intent="click the login button",
        encoder_spec=EncoderSpec(),
        viewport=Viewport(width=1920, height=1080),
        client_capabilities=[],
        created_at=time.time(),
        last_seen_at=time.time(),
    )


@pytest.mark.asyncio
async def test_pipeline_retries_on_bad_json():
    """Pipeline should retry and succeed after first bad response."""
    fake_tensor = torch.zeros(576, 1152)
    planner = FakePlanner(
        responses=[
            "oops, here's my answer: {invalid json",
            {"action": "CLICK", "confidence": 0.8, "coords": [0.5, 0.5]},
        ]
    )
    pipeline = ActionPlanningPipeline(planner)
    frame = _make_frame()
    session = _make_session()
    ctx = pipeline.build_context(session, frame, fake_tensor)
    directive = await pipeline.plan(ctx)
    assert directive.action.value == "CLICK"


@pytest.mark.asyncio
async def test_pipeline_raises_after_max_retries():
    """Pipeline raises RuntimeError after all retries fail."""
    fake_tensor = torch.zeros(576, 1152)
    planner = FakePlanner(responses=["bad response"] * 5)
    pipeline = ActionPlanningPipeline(planner)
    frame = _make_frame()
    session = _make_session()
    ctx = pipeline.build_context(session, frame, fake_tensor)
    with pytest.raises(RuntimeError, match="failed after"):
        await pipeline.plan(ctx)


@pytest.mark.asyncio
async def test_pipeline_passes_score_grid_to_planner():
    """Pipeline decodes and passes score_grid correctly."""
    import base64, struct

    # Build a score_grid: highest at patch 300
    scores = [0.0] * 576
    scores[300] = 1.0
    raw = struct.pack("<576f", *scores)
    encoded = base64.b64encode(raw).decode()

    from app.schemas.messages import ScoreGrid

    captured: dict = {}

    class CapturePlanner:
        async def plan_action(self, **kwargs):
            captured.update(kwargs)
            return {"action": "CLICK", "confidence": 0.9, "coords": [0.6, 0.6]}

        async def health_check(self):
            return True

    pipeline = ActionPlanningPipeline(CapturePlanner())
    frame = _make_frame()
    frame.score_grid = ScoreGrid(data=encoded, shape=[24, 24])
    session = _make_session()
    fake_tensor = torch.zeros(576, 1152)
    ctx = pipeline.build_context(session, frame, fake_tensor)
    await pipeline.plan(ctx)
    assert ctx.score_grid is not None
    assert ctx.score_grid[300] == pytest.approx(1.0, abs=1e-5)
