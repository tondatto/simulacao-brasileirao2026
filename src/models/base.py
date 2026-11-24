"""
Interface base e utilitários compartilhados pelos modelos de projeção de partidas.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import poisson as scipy_poisson

MAX_GOALS = 8


def build_score_matrix(
    lam_home: float,
    lam_away: float,
    rho: float = 0.0,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Constrói a matriz de probabilidades conjuntas de placares (0..max_goals)
    usando distribuição de Poisson independente com correção opcional de Dixon-Coles (rho).
    Linhas representam gols do mandante; colunas, gols do visitante.
    """
    goals = np.arange(max_goals + 1)
    p_home = scipy_poisson.pmf(goals, max(1e-6, lam_home))
    p_away = scipy_poisson.pmf(goals, max(1e-6, lam_away))
    mat = np.outer(p_home, p_away).astype(np.float64)

    if rho != 0.0:
        mat[0, 0] *= max(0.0, 1.0 - lam_home * lam_away * rho)
        mat[0, 1] *= max(0.0, 1.0 + lam_home * rho)
        mat[1, 0] *= max(0.0, 1.0 + lam_away * rho)
        mat[1, 1] *= max(0.0, 1.0 - rho)

    mat = np.clip(mat, 0.0, None)
    total = float(mat.sum())
    if total <= 0.0:
        mat = np.outer(p_home, p_away).astype(np.float64)
        total = float(mat.sum())
    return mat / total


def probs_1x2_from_score_matrix(score_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Extrai (p_mandante, p_empate, p_visitante) de uma matriz de placares M[h, a].
    """
    pm = float(np.tril(score_matrix, k=-1).sum())
    pe = float(np.trace(score_matrix))
    pv = float(np.triu(score_matrix, k=1).sum())
    total = pm + pe + pv
    if total <= 0.0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return pm / total, pe / total, pv / total


def apply_zebra_1x2(pm: float, pe: float, pv: float, fator_zebra: float) -> Tuple[float, float, float]:
    """
    Aplica inversão estocástica (fator de zebra) sobre probabilidades 1X2 e normaliza.
    """
    if fator_zebra > 0.0:
        pm_z = (1.0 - fator_zebra) * pm + fator_zebra * pv
        pv_z = (1.0 - fator_zebra) * pv + fator_zebra * pm
        pm, pv = pm_z, pv_z
    total = pm + pe + pv
    return pm / total, pe / total, pv / total


def apply_zebra_score_matrix(score_matrix: np.ndarray, fator_zebra: float) -> np.ndarray:
    """
    Aplica inversão estocástica equivalente diretamente sobre a matriz de placares.
    """
    if fator_zebra <= 0.0:
        total = float(score_matrix.sum())
        return score_matrix / total if total > 0.0 else score_matrix
    mat = (1.0 - fator_zebra) * score_matrix + fator_zebra * score_matrix.T
    mat = np.clip(mat, 0.0, None)
    return mat / float(mat.sum())


class MatchModel:
    """Classe base para modelos plugáveis de projeção de partidas."""

    name: str = "base"
    description: str = "Modelo Base"
    requires_history: bool = False
    uses_score_simulation: bool = False

    def prepare_matches(
        self,
        standings: List[Dict[str, Any]],
        matches: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fator_zebra: float = 0.05,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError
