"""Deterministic mock planner for local development and testing.

Task 6 update: accepts score_grid (list[float] | None) and layout_summary.
The score_grid is used to pick the highest-scoring non-sensitive patch for CLICK.
The 'click highest-norm patch' heuristic is removed from the default code path —
the planner now uses the score_grid when available.
"""

from typing import Any

from app.models.base import ActionPlanner
from app.planning.representation import SpatialReasoningFeatures


class MockActionPlanner(ActionPlanner):
    async def plan_action(
        self,
        *,
        task_intent: str,
        spatial_features: SpatialReasoningFeatures,
        state_version: int,
        frame_id: str,
        last_action_id: str | None,
        score_grid: list[float] | None = None,
        layout_summary: str | None = None,
    ) -> dict[str, Any]:
        intent_lower = task_intent.lower()

        if "scroll" in intent_lower:
            return {
                "action": "SCROLL",
                "scroll_delta": [0.0, 300.0],
                "confidence": 0.75,
                "reasoning_summary": "Task intent mentions scrolling; issuing downward scroll.",
            }

        if "type" in intent_lower or "enter" in intent_lower or "input" in intent_lower:
            # Use score_grid to find best non-sensitive patch for typing target
            target_idx, x, y = self._best_patch(spatial_features, score_grid)
            return {
                "action": "TYPE",
                "coords": [x, y],
                "text": "example_input",
                "target_id": target_idx,
                "confidence": 0.8,
                "reasoning_summary": (
                    f"Typing at patch {target_idx} ({x:.2f}, {y:.2f}) "
                    f"selected via {'score_grid' if score_grid else 'spatial_features'}."
                ),
            }

        if "navigate" in intent_lower or "http" in intent_lower or "url" in intent_lower:
            return {
                "action": "NAVIGATE",
                "text": "https://example.com",
                "confidence": 0.7,
                "reasoning_summary": "Navigation intent detected.",
            }

        if last_action_id and state_version > 2:
            return {
                "action": "DONE",
                "confidence": 0.9,
                "reasoning_summary": "Follow-up frame after prior action; marking task complete.",
            }

        # Default: CLICK at best patch per score_grid (or spatial features fallback)
        target_idx, x, y = self._best_patch(spatial_features, score_grid)
        return {
            "action": "CLICK",
            "coords": [x, y],
            "target_id": target_idx,
            "confidence": 0.85,
            "reasoning_summary": (
                f"Click at patch {target_idx} ({x:.2f}, {y:.2f}) "
                f"selected via {'score_grid' if score_grid else 'spatial_features'}."
            ),
        }

    def _best_patch(
        self,
        spatial_features: SpatialReasoningFeatures,
        score_grid: list[float] | None,
    ) -> tuple[int, float, float]:
        """Pick target patch index and normalised coords.

        When score_grid is available, picks the patch with the highest score.
        Falls back to dominant_patch_index from spatial features.
        """
        if score_grid and len(score_grid) == spatial_features.patch_count:
            best_idx = int(max(range(len(score_grid)), key=lambda i: score_grid[i]))
        else:
            best_idx = spatial_features.dominant_patch_index

        grid_cols = spatial_features.grid_cols
        grid_rows = spatial_features.grid_rows
        col = best_idx % grid_cols
        row = best_idx // grid_cols
        x = (col + 0.5) / grid_cols
        y = (row + 0.5) / grid_rows
        return best_idx, x, y

    async def health_check(self) -> bool:
        return True
