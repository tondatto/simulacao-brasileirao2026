"""
Wrapper do modelo baseline Prof. Rogemar Riffel (IF-UFRGS) na interface MatchModel.
"""

from typing import Any, Dict, List, Optional

from src.model_riffel import calculate_team_strengths, prepare_matches_with_probabilities
from src.models.base import MatchModel


class RiffelModel(MatchModel):
    name = "riffel"
    description = "Prof. Rogemar Riffel (50% Histórico + 50% Forma Recente)"
    requires_history = False
    uses_score_simulation = False

    def prepare_matches(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fator_zebra: float = 0.05,
    ) -> List[Dict[str, Any]]:
        strengths = calculate_team_strengths(standings)
        return prepare_matches_with_probabilities(matches, strengths, fator_zebra=fator_zebra)
