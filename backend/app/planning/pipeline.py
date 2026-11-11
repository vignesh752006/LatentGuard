import uuid
from dataclasses import dataclass

import torch

from app.planning.json_repair import PlannerResponseError, repair_json
from app.planning.representation import SpatialReasoningFeatures, extract_spatial_features
from app.schemas.messages import (
    ActionDirectivePayload,
    ActionType,
    LatentFramePayload,
    ScoreGrid,
)
from app.session.browser_state import BrowserSession
from app.validation.action_schema import validate_action_directive

import base64
import struct
import logging

logger = logging.getLogger(__name__)

MAX_PLANNER_RETRIES = 3


@dataclass
class PlanningContext:
    session: BrowserSession
    frame: LatentFramePayload
    tensor: torch.Tensor
    spatial_features: SpatialReasoningFeatures
    score_grid: list[float] | None


def _decode_score_grid(sg: ScoreGrid | None) -> list[float] | None:
    """Decode base64 Float32 score grid into a Python list."""
    if sg is None:
        return None
    try:
        raw = base64.b64decode(sg.data)
        n = len(raw) // 4
        return list(struct.unpack(f"<{n}f", raw[:n * 4]))
    except Exception as exc:
        logger.warning("Failed to decode score_grid: %s", exc)
        return None


class ActionPlanningPipeline:
    def __init__(self, planner):
        self._planner = planner

    def build_context(
        self,
        session: BrowserSession,
        frame: LatentFramePayload,
        tensor: torch.Tensor,
    ) -> PlanningContext:
        grid = frame.spatial_metadata.patch_grid
        features = extract_spatial_features(
            tensor,
            grid_rows=grid.rows,
            grid_cols=grid.cols,
            sensitive_patch_indices=frame.spatial_metadata.sensitive_patch_indices,
        )
        return PlanningContext(
            session=session,
            frame=frame,
            tensor=tensor,
            spatial_features=features,
            score_grid=_decode_score_grid(frame.score_grid),
        )

    async def plan(self, context: PlanningContext) -> ActionDirectivePayload:
        last_error: Exception | None = None

        for attempt in range(1, MAX_PLANNER_RETRIES + 1):
            try:
                raw = await self._planner.plan_action(
                    task_intent=context.session.task_intent,
                    spatial_features=context.spatial_features,
                    state_version=context.frame.state_version,
                    frame_id=context.frame.frame_id,
                    last_action_id=context.frame.action_context.last_action_id,
                    score_grid=context.score_grid,
                )
                # If planner returns a string, attempt JSON repair
                if isinstance(raw, str):
                    raw = repair_json(raw)

                directive = ActionDirectivePayload(
                    action_id=str(uuid.uuid4()),
                    action=ActionType(raw["action"]),
                    target_id=raw.get("target_id"),
                    coords=raw.get("coords"),
                    text=raw.get("text"),
                    scroll_delta=raw.get("scroll_delta"),
                    state_version_required=context.frame.state_version,
                    confidence=float(raw.get("confidence", 0.5)),
                    reasoning_summary=raw.get("reasoning_summary", ""),
                )
                return validate_action_directive(directive)

            except (PlannerResponseError, KeyError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Planner attempt %d/%d failed: %s",
                    attempt,
                    MAX_PLANNER_RETRIES,
                    exc,
                )

        raise RuntimeError(
            f"Planner failed after {MAX_PLANNER_RETRIES} retries. "
            f"Last error: {last_error}"
        )
