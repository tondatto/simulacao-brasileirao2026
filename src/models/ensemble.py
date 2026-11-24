"""
Modelo Ensemble combinando os modelos Riffel, Poisson, Elo e Dixon-Coles.
"""

from typing import Any, Dict, List, Optional

from src.models.base import MatchModel
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloModel
from src.models.poisson import PoissonModel
from src.models.riffel import RiffelModel


class EnsembleModel(MatchModel):
    name = "ensemble"
    description = "Ensemble Ponderado (Riffel + Poisson + Elo + Dixon-Coles)"
    requires_history = True
    uses_score_simulation = False

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.models: List[MatchModel] = [
            RiffelModel(),
            PoissonModel(),
            EloModel(),
            DixonColesModel(),
        ]
        self.weights = weights or {
            "riffel": 0.25,
            "poisson": 0.25,
            "elo": 0.25,
            "dixon-coles": 0.25,
        }

    def prepare_matches(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fator_zebra: float = 0.05,
    ) -> List[Dict[str, Any]]:
        if not matches:
            return []

        per_model_matches = [
            model.prepare_matches(standings, matches, history=history, fator_zebra=fator_zebra)
            for model in self.models
        ]
        model_weights = [float(self.weights.get(model.name, 1.0)) for model in self.models]

        enriched: List[Dict[str, Any]] = []
        for idx, base_match in enumerate(matches):
            pm = sum(w * per_model_matches[m_i][idx]["p_mandante"] for m_i, w in enumerate(model_weights))
            pe = sum(w * per_model_matches[m_i][idx]["p_empate"] for m_i, w in enumerate(model_weights))
            pv = sum(w * per_model_matches[m_i][idx]["p_visitante"] for m_i, w in enumerate(model_weights))
            total = pm + pe + pv
            enriched.append({
                **base_match,
                "p_mandante": pm / total,
                "p_empate": pe / total,
                "p_visitante": pv / total,
            })
        return enriched
