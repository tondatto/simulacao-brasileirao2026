"""
Análise individual de permanência na Série A (luta contra o rebaixamento / Z-4).
Calcula combinações de resultados, metas de pontuação e jogos-chave de rivais
de forma compatível com modelos 1X2 e modelos baseados em placar exato (score_matrix).
"""

import itertools
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).strip().lower()


def resolve_team_name(query: str, standings: List[Dict[str, Any]]) -> str:
    """
    Resolve nome ou sigla de clube de forma tolerante a maiúsculas/minúsculas e acentos.
    Lança ValueError se a consulta não existir ou for ambígua.
    """
    if not query or not query.strip():
        raise ValueError("Informe o nome ou a sigla de um clube para análise.")

    q = _normalize_text(query)

    # 1. Correspondência exata por nome normalizado
    for t in standings:
        if _normalize_text(t["nome"]) == q:
            return t["nome"]

    # 2. Correspondência exata por sigla normalizada
    for t in standings:
        if _normalize_text(t.get("sigla", "")) == q:
            return t["nome"]

    # 3. Substring única no nome normalizado
    matches = [t["nome"] for t in standings if q in _normalize_text(t["nome"])]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Consulta '{query}' é ambígua entre: {', '.join(matches)}.")

    disponiveis = ", ".join(
        f"{t['nome']} ({t.get('sigla', '')})" if t.get("sigla") else t["nome"]
        for t in standings
    )
    raise ValueError(f"Time '{query}' não encontrado. Clubes disponíveis: {disponiveis}")


def _apply_random_matches(
    pts: np.ndarray,
    wins: np.ndarray,
    gd: np.ndarray,
    gp: np.ndarray,
    matches: List[Dict[str, Any]],
    team_to_idx: Dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simula uma lista de partidas atualizando pts/wins/gd/gp in-place.
    Retorna matriz (N, len(matches)) com 0=Mandante, 1=Empate, 2=Visitante.
    """
    N = pts.shape[0]
    num_matches = len(matches)
    outcomes = np.empty((N, num_matches), dtype=np.int8)
    if num_matches == 0:
        return outcomes

    rand_vals = rng.random((N, num_matches), dtype=np.float32)
    needs_margin = any(m.get("score_matrix") is None for m in matches)
    rand_margin = (
        rng.choice([1, 2, 3], size=(N, num_matches), p=[0.55, 0.30, 0.15])
        if needs_margin
        else None
    )

    for k, match in enumerate(matches):
        h_idx = team_to_idx[match["mandante"]]
        a_idx = team_to_idx[match["visitante"]]
        r = rand_vals[:, k]

        if match.get("score_matrix") is not None:
            mat = np.asarray(match["score_matrix"], dtype=np.float64)
            dim = mat.shape[0]
            cdf = np.cumsum(mat.ravel())
            cdf[-1] = 1.0
            idx = np.minimum(np.searchsorted(cdf, r, side="right"), dim * dim - 1)
            h_goals = idx // dim
            a_goals = idx % dim

            home_win = h_goals > a_goals
            draw = h_goals == a_goals
            away_win = a_goals > h_goals

            pts[:, h_idx] += home_win * 3 + draw
            pts[:, a_idx] += away_win * 3 + draw
            wins[:, h_idx] += home_win
            wins[:, a_idx] += away_win
            gd[:, h_idx] += h_goals - a_goals
            gd[:, a_idx] += a_goals - h_goals
            gp[:, h_idx] += h_goals
            gp[:, a_idx] += a_goals
        else:
            pm = match["p_mandante"]
            pe = match["p_empate"]
            t1 = pm
            t2 = pm + pe
            margin = rand_margin[:, k]

            home_win = r < t1
            draw = (r >= t1) & (r < t2)
            away_win = r >= t2

            pts[:, h_idx] += home_win * 3 + draw
            pts[:, a_idx] += away_win * 3 + draw
            wins[:, h_idx] += home_win
            wins[:, a_idx] += away_win

            h_margin = home_win * margin
            a_margin = away_win * margin
            gd[:, h_idx] += h_margin - a_margin
            gd[:, a_idx] += a_margin - h_margin
            gp[:, h_idx] += h_margin + draw
            gp[:, a_idx] += a_margin + draw

        outcomes[:, k] = np.where(home_win, 0, np.where(draw, 1, 2))

    return outcomes


def _apply_fixed_match(
    pts: np.ndarray,
    wins: np.ndarray,
    gd: np.ndarray,
    gp: np.ndarray,
    match: Dict[str, Any],
    team_to_idx: Dict[str, int],
    target_is_home: bool,
    outcome: str,
    rng: np.random.Generator,
) -> None:
    """
    Aplica um resultado fixo ('V', 'E', 'D' sob a ótica do time analisado)
    atualizando tanto o time analisado quanto o adversário.
    """
    N = pts.shape[0]
    h_idx = team_to_idx[match["mandante"]]
    a_idx = team_to_idx[match["visitante"]]

    if outcome == "E":
        match_outcome = "D"
    elif (outcome == "V" and target_is_home) or (outcome == "D" and not target_is_home):
        match_outcome = "H"
    else:
        match_outcome = "A"

    if match.get("score_matrix") is not None:
        mat = np.asarray(match["score_matrix"], dtype=np.float64)
        dim = mat.shape[0]
        if match_outcome == "H":
            sub = np.tril(mat, k=-1)
        elif match_outcome == "D":
            sub = np.diag(np.diag(mat))
        else:
            sub = np.triu(mat, k=1)

        mass = float(sub.sum())
        if mass <= 1e-12:
            if match_outcome == "H":
                h_goals = np.ones(N, dtype=np.int64)
                a_goals = np.zeros(N, dtype=np.int64)
            elif match_outcome == "D":
                h_goals = np.zeros(N, dtype=np.int64)
                a_goals = np.zeros(N, dtype=np.int64)
            else:
                h_goals = np.zeros(N, dtype=np.int64)
                a_goals = np.ones(N, dtype=np.int64)
        else:
            cdf = np.cumsum((sub / mass).ravel())
            cdf[-1] = 1.0
            r = rng.random(N, dtype=np.float32)
            idx = np.minimum(np.searchsorted(cdf, r, side="right"), dim * dim - 1)
            h_goals = idx // dim
            a_goals = idx % dim

        if match_outcome == "H":
            pts[:, h_idx] += 3
            wins[:, h_idx] += 1
        elif match_outcome == "A":
            pts[:, a_idx] += 3
            wins[:, a_idx] += 1
        else:
            pts[:, h_idx] += 1
            pts[:, a_idx] += 1

        gd[:, h_idx] += h_goals - a_goals
        gd[:, a_idx] += a_goals - h_goals
        gp[:, h_idx] += h_goals
        gp[:, a_idx] += a_goals
        return

    if match_outcome == "H":
        margin = rng.choice([1, 2, 3], size=N, p=[0.55, 0.30, 0.15])
        pts[:, h_idx] += 3
        wins[:, h_idx] += 1
        gd[:, h_idx] += margin
        gd[:, a_idx] -= margin
        gp[:, h_idx] += margin
    elif match_outcome == "A":
        margin = rng.choice([1, 2, 3], size=N, p=[0.55, 0.30, 0.15])
        pts[:, a_idx] += 3
        wins[:, a_idx] += 1
        gd[:, a_idx] += margin
        gd[:, h_idx] -= margin
        gp[:, a_idx] += margin
    else:
        pts[:, h_idx] += 1
        pts[:, a_idx] += 1
        gp[:, h_idx] += 1
        gp[:, a_idx] += 1


def _compute_survival_mask(
    pts: np.ndarray,
    wins: np.ndarray,
    gd: np.ndarray,
    gp: np.ndarray,
    target_idx: int,
    safe_max_pos: int,
    tie_noise: np.ndarray,
) -> np.ndarray:
    """
    Retorna máscara booleana (N,) indicando simulações em que o time termina fora do Z-4.
    """
    score = (
        pts * 10_000_000_000
        + wins * 10_000_000
        + (gd + 500) * 10_000
        + (gp + 500) * 100
        + tie_noise
    )
    better = np.sum(score > score[:, target_idx : target_idx + 1], axis=1)
    return better < safe_max_pos


def _outcome_prob_for_team(match: Dict[str, Any], target_is_home: bool, outcome: str) -> float:
    if outcome == "E":
        return float(match["p_empate"])
    if (outcome == "V" and target_is_home) or (outcome == "D" and not target_is_home):
        return float(match["p_mandante"])
    return float(match["p_visitante"])


def _format_combo_label(combo: Tuple[str, ...], team_matches_info: List[Dict[str, Any]]) -> str:
    if not combo:
        return "Sem jogos restantes"
    parts = []
    for res, info in zip(combo, team_matches_info):
        sigla = info["adversario_sigla"] or info["adversario"][:3].upper()
        parts.append(f"{res}({sigla})")
    return " · ".join(parts)


def _format_combo_summary(combo: Tuple[str, ...]) -> str:
    v = sum(1 for r in combo if r == "V")
    e = sum(1 for r in combo if r == "E")
    d = sum(1 for r in combo if r == "D")
    return f"{v}V {e}E {d}D"


def _analyze_rival_matches(
    other_matches: List[Dict[str, Any]],
    other_outcomes: np.ndarray,
    survival_mask: np.ndarray,
    conditioned_label: str,
    team_pos_map: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], str]:
    if not other_matches or other_outcomes.shape[1] == 0:
        return [], conditioned_label

    key_matches: List[Dict[str, Any]] = []
    for k, match in enumerate(other_matches):
        col = other_outcomes[:, k]
        outcome_probs: Dict[int, float] = {}
        for code in (0, 1, 2):
            mask = col == code
            if int(np.sum(mask)) >= 30:
                outcome_probs[code] = float(np.mean(survival_mask[mask])) * 100.0

        if len(outcome_probs) < 2:
            continue

        best_code = max(outcome_probs, key=lambda c: outcome_probs[c])
        worst_code = min(outcome_probs, key=lambda c: outcome_probs[c])
        impact_pp = outcome_probs[best_code] - outcome_probs[worst_code]
        if impact_pp < 0.5:
            continue

        if best_code == 0:
            melhor_resultado = f"Vitória do {match['mandante']}"
        elif best_code == 1:
            melhor_resultado = "Empate"
        else:
            melhor_resultado = f"Vitória do {match['visitante']}"

        key_matches.append({
            "rodada": int(match.get("rodada") or 0),
            "mandante": match["mandante"],
            "mandante_pos": team_pos_map.get(match["mandante"], 0),
            "visitante": match["visitante"],
            "visitante_pos": team_pos_map.get(match["visitante"], 0),
            "melhor_resultado": melhor_resultado,
            "prob_melhor_pct": outcome_probs[best_code],
            "prob_pior_pct": outcome_probs[worst_code],
            "impacto_pp": impact_pp,
            "p_escapar_mandante": outcome_probs.get(0),
            "p_escapar_empate": outcome_probs.get(1),
            "p_escapar_visitante": outcome_probs.get(2),
        })

    key_matches.sort(key=lambda x: (-x["impacto_pp"], x["rodada"]))
    return key_matches[:5], conditioned_label


def _build_recommended_target(
    combinations: List[Dict[str, Any]],
    points_summary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    safe_pts_list = [p["pontos_finais"] for p in points_summary if p["prob_escapar_pct"] >= 99.9]
    likely_pts_list = [p["pontos_finais"] for p in points_summary if p["prob_escapar_pct"] >= 50.0]

    safe_points = min(safe_pts_list) if safe_pts_list else None
    likely_points = min(likely_pts_list) if likely_pts_list else None

    easiest_safe = None
    easiest_likely = None
    best_combo = combinations[0] if combinations else None

    if combinations:
        safe_combos = [c for c in combinations if c["prob_escapar_pct"] >= 99.9]
        if safe_combos:
            easiest_safe = min(
                safe_combos,
                key=lambda c: (c["pontos_finais"], -c["prob_cenario_pct"], -c["prob_escapar_pct"]),
            )
        likely_combos = [c for c in combinations if c["prob_escapar_pct"] >= 50.0]
        if likely_combos:
            easiest_likely = min(
                likely_combos,
                key=lambda c: (c["pontos_finais"], -c["prob_cenario_pct"], -c["prob_escapar_pct"]),
            )

    return {
        "safe_points": safe_points,
        "likely_points": likely_points,
        "easiest_safe_combo": easiest_safe,
        "easiest_likely_combo": easiest_likely,
        "best_combo": best_combo,
    }


def analyze_team_relegation(
    standings: List[Dict[str, Any]],
    enriched_matches: List[Dict[str, Any]],
    team_name: str,
    num_simulations: int = 100_000,
    seed: int = 42,
    relegation_slots: int = 4,
    max_enum_matches: int = 5,
) -> Dict[str, Any]:
    """
    Analisa detalhadamente os cenários de permanência de um clube na Série A.
    """
    team_names = [t["nome"] for t in standings]
    num_teams = len(team_names)
    team_to_idx = {name: i for i, name in enumerate(team_names)}
    if team_name not in team_to_idx:
        raise ValueError(f"Time '{team_name}' não encontrado na classificação.")

    target_idx = team_to_idx[team_name]
    target_team = standings[target_idx]
    team_pos_map = {t["nome"]: int(t["posicao"]) for t in standings}
    team_sigla_map = {t["nome"]: t.get("sigla", "") for t in standings}
    safe_max_pos = max(1, num_teams - relegation_slots)
    direct_rival_cutoff = max(1, num_teams - relegation_slots - 3)

    team_matches: List[Dict[str, Any]] = []
    other_matches: List[Dict[str, Any]] = []
    for m in enriched_matches:
        if m["mandante"] == team_name or m["visitante"] == team_name:
            team_matches.append(m)
        else:
            other_matches.append(m)

    team_matches.sort(key=lambda m: int(m.get("rodada") or 0))
    K = len(team_matches)

    team_matches_info: List[Dict[str, Any]] = []
    for m in team_matches:
        is_home = m["mandante"] == team_name
        adv = m["visitante"] if is_home else m["mandante"]
        adv_pos = team_pos_map.get(adv, 0)
        team_matches_info.append({
            "rodada": int(m.get("rodada") or 0),
            "mando": "Casa" if is_home else "Fora",
            "target_is_home": is_home,
            "adversario": adv,
            "adversario_sigla": team_sigla_map.get(adv, ""),
            "adversario_pos": adv_pos,
            "confronto_direto": adv_pos >= direct_rival_cutoff,
            "p_vitoria": _outcome_prob_for_team(m, is_home, "V") * 100.0,
            "p_empate": _outcome_prob_for_team(m, is_home, "E") * 100.0,
            "p_derrota": _outcome_prob_for_team(m, is_home, "D") * 100.0,
        })

    if K <= 3:
        N = max(1, min(int(num_simulations), 100_000))
    elif K <= max_enum_matches:
        N = max(1, min(int(num_simulations), 20_000))
    else:
        N = max(1, min(int(num_simulations), 100_000))

    init_pts = np.array([t["pontos"] for t in standings], dtype=np.int64)
    init_wins = np.array([t["vitorias"] for t in standings], dtype=np.int64)
    init_gd = np.array([t["saldo_gols"] for t in standings], dtype=np.int64)
    init_gp = np.array([t["gols_pro"] for t in standings], dtype=np.int64)

    base_pts = np.tile(init_pts, (N, 1))
    base_wins = np.tile(init_wins, (N, 1))
    base_gd = np.tile(init_gd, (N, 1))
    base_gp = np.tile(init_gp, (N, 1))

    rng = np.random.default_rng(seed)
    other_outcomes = _apply_random_matches(base_pts, base_wins, base_gd, base_gp, other_matches, team_to_idx, rng)
    tie_noise = rng.integers(0, 99, size=(N, num_teams), dtype=np.int64)

    # Simulação não-condicionada dos jogos do time (usada no fallback e na análise de rivais)
    uncond_pts = base_pts.copy()
    uncond_wins = base_wins.copy()
    uncond_gd = base_gd.copy()
    uncond_gp = base_gp.copy()
    _apply_random_matches(uncond_pts, uncond_wins, uncond_gd, uncond_gp, team_matches, team_to_idx, rng)
    uncond_survived = _compute_survival_mask(
        uncond_pts, uncond_wins, uncond_gd, uncond_gp, target_idx, safe_max_pos, tie_noise
    )

    enumerated = K <= max_enum_matches
    combinations: List[Dict[str, Any]] = []
    points_summary: List[Dict[str, Any]] = []
    best_combo_survived: Optional[np.ndarray] = None

    if enumerated:
        best_survival = -1.0
        worst_survival = 101.0

        for combo in itertools.product(("V", "E", "D"), repeat=K):
            sim_pts = base_pts.copy()
            sim_wins = base_wins.copy()
            sim_gd = base_gd.copy()
            sim_gp = base_gp.copy()

            prob_cenario = 1.0
            for j, res in enumerate(combo):
                is_home = team_matches_info[j]["target_is_home"]
                prob_cenario *= _outcome_prob_for_team(team_matches[j], is_home, res)
                _apply_fixed_match(sim_pts, sim_wins, sim_gd, sim_gp, team_matches[j], team_to_idx, is_home, res, rng)

            survived = _compute_survival_mask(
                sim_pts, sim_wins, sim_gd, sim_gp, target_idx, safe_max_pos, tie_noise
            )
            prob_escapar_pct = float(np.mean(survived)) * 100.0

            if combo == ("V",) * K:
                best_combo_survived = survived
                best_survival = prob_escapar_pct
            if combo == ("D",) * K:
                worst_survival = prob_escapar_pct

            pts_ganhos = sum(3 if r == "V" else (1 if r == "E" else 0) for r in combo)
            v_ganhas = sum(1 for r in combo if r == "V")

            combinations.append({
                "combo": combo,
                "resumo": _format_combo_summary(combo),
                "resultados_str": _format_combo_label(combo, team_matches_info),
                "pontos_ganhos": pts_ganhos,
                "pontos_finais": int(target_team["pontos"]) + pts_ganhos,
                "vitorias_ganhas": v_ganhas,
                "prob_cenario_pct": prob_cenario * 100.0,
                "prob_escapar_pct": prob_escapar_pct,
                "prob_z4_pct": 100.0 - prob_escapar_pct,
            })

        combinations.sort(key=lambda c: (-c["prob_escapar_pct"], -c["pontos_finais"], -c["prob_cenario_pct"]))

        total_weight = sum(c["prob_cenario_pct"] for c in combinations)
        if total_weight > 0:
            base_survival_pct = sum(c["prob_cenario_pct"] * c["prob_escapar_pct"] for c in combinations) / total_weight
        else:
            base_survival_pct = float(np.mean(uncond_survived)) * 100.0

        by_pts: Dict[int, List[Dict[str, Any]]] = {}
        for c in combinations:
            by_pts.setdefault(c["pontos_finais"], []).append(c)

        for pf in sorted(by_pts.keys(), reverse=True):
            group = by_pts[pf]
            prob_atingir = sum(c["prob_cenario_pct"] for c in group)
            if prob_atingir > 1e-12:
                prob_escapar = sum(c["prob_cenario_pct"] * c["prob_escapar_pct"] for c in group) / prob_atingir
            else:
                prob_escapar = float(np.mean([c["prob_escapar_pct"] for c in group]))
            exemplos = ", ".join(sorted({c["resumo"] for c in group}))
            points_summary.append({
                "pontos_finais": pf,
                "pontos_ganhos": pf - int(target_team["pontos"]),
                "prob_atingir_pct": prob_atingir,
                "prob_escapar_pct": prob_escapar,
                "exemplos": exemplos,
            })
    else:
        base_survival_pct = float(np.mean(uncond_survived)) * 100.0

        def _eval_extreme(fixed_res: str) -> Tuple[float, np.ndarray]:
            sim_pts = base_pts.copy()
            sim_wins = base_wins.copy()
            sim_gd = base_gd.copy()
            sim_gp = base_gp.copy()
            for j in range(K):
                _apply_fixed_match(
                    sim_pts, sim_wins, sim_gd, sim_gp,
                    team_matches[j], team_to_idx, team_matches_info[j]["target_is_home"], fixed_res, rng
                )
            mask = _compute_survival_mask(
                sim_pts, sim_wins, sim_gd, sim_gp, target_idx, safe_max_pos, tie_noise
            )
            return float(np.mean(mask)) * 100.0, mask

        best_survival, best_combo_survived = _eval_extreme("V")
        worst_survival, _ = _eval_extreme("D")

        final_pts = uncond_pts[:, target_idx]
        for pf in sorted(np.unique(final_pts).tolist(), reverse=True):
            mask = final_pts == pf
            cnt = int(np.sum(mask))
            points_summary.append({
                "pontos_finais": int(pf),
                "pontos_ganhos": int(pf) - int(target_team["pontos"]),
                "prob_atingir_pct": (cnt / N) * 100.0,
                "prob_escapar_pct": float(np.mean(uncond_survived[mask])) * 100.0,
                "exemplos": f"+{int(pf) - int(target_team['pontos'])} pts",
            })

    base_relegation_pct = 100.0 - base_survival_pct

    if worst_survival >= 99.95:
        status = "ja_salvo"
    elif best_survival <= 0.05:
        status = "ja_rebaixado"
    elif best_survival >= 99.95:
        status = "depende_so_de_si"
    else:
        status = "depende_de_rivais"

    if status in ("ja_salvo", "ja_rebaixado") or not other_matches:
        key_rival_matches: List[Dict[str, Any]] = []
        rival_conditioned_on = "não aplicável"
    else:
        if 5.0 <= base_survival_pct <= 95.0:
            ref_mask = uncond_survived
            cond_label = "simulação geral"
        elif base_survival_pct < 5.0 and best_combo_survived is not None:
            ref_mask = best_combo_survived
            cond_label = "vencendo todos os seus jogos"
        else:
            ref_mask = uncond_survived
            cond_label = "simulação geral"

        key_rival_matches, rival_conditioned_on = _analyze_rival_matches(
            other_matches, other_outcomes, ref_mask, cond_label, team_pos_map
        )

    recommended_target = _build_recommended_target(combinations, points_summary)

    return {
        "team": {
            "nome": target_team["nome"],
            "sigla": target_team.get("sigla", ""),
            "pos_atual": int(target_team["posicao"]),
            "pontos_atuais": int(target_team["pontos"]),
            "vitorias_atuais": int(target_team["vitorias"]),
            "saldo_atual": int(target_team["saldo_gols"]),
            "jogos_atuais": int(target_team.get("jogos", 0)),
            "jogos_restantes": K,
            "max_pontos_possiveis": int(target_team["pontos"]) + 3 * K,
        },
        "num_simulations": N,
        "status": status,
        "base_survival_pct": base_survival_pct,
        "base_relegation_pct": base_relegation_pct,
        "best_survival_pct": best_survival,
        "worst_survival_pct": worst_survival,
        "enumerated": enumerated,
        "team_matches": team_matches_info,
        "combinations": combinations,
        "points_summary": points_summary,
        "key_rival_matches": key_rival_matches,
        "rival_conditioned_on": rival_conditioned_on,
        "recommended_target": recommended_target,
    }
