"""
Registro central de metodologias de projeção de partidas.
"""

from typing import Dict, Type

from src.models.base import MatchModel
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloModel
from src.models.ensemble import EnsembleModel
from src.models.poisson import PoissonModel
from src.models.riffel import RiffelModel

MODEL_REGISTRY: Dict[str, Type[MatchModel]] = {
    "riffel": RiffelModel,
    "poisson": PoissonModel,
    "elo": EloModel,
    "dixon-coles": DixonColesModel,
    "ensemble": EnsembleModel,
}


def get_model(name: str) -> MatchModel:
    if name not in MODEL_REGISTRY:
        disponiveis = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo '{name}' desconhecido. Opções disponíveis: {disponiveis}")
    return MODEL_REGISTRY[name]()
