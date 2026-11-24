"""
Módulo para extração de dados do Campeonato Brasileiro 2026 diretamente do Globo Esporte (GE),
baseado na arquitetura do repositório ezefranca/campeonato-brasileiro-api.
Inclui cache local em disco em .cache/ para execução offline e máxima performance.
"""

import json
import os
import re
import urllib.request
from typing import Dict, Any, List, Tuple

GE_SERIE_A_URL = "https://ge.globo.com/futebol/brasileirao-serie-a/"
DEFAULT_RESOURCE_ID = "d1a37fa4-e948-43a6-ba53-ab24ab3a45b1"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _request_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _extract_resource_id(html: str) -> str:
    m = re.search(r'data-bs-resource-id="([a-f0-9-]+)"', html)
    if m:
        return m.group(1)
    return DEFAULT_RESOURCE_ID


def _extract_classificacao_payload(html: str) -> Dict[str, Any]:
    # Procura pelo bloco JavaScript onde os dados do React estão embutidos
    m = re.search(r'const\s+classificacao\s*=\s*(\{.*?\});\s*(?:const|</script>)', html, re.DOTALL)
    if not m:
        m = re.search(r'const\s+classificacao\s*=\s*(\{.*?\});', html, re.DOTALL)
    if not m:
        raise ValueError("Não foi possível encontrar o objeto 'classificacao' no HTML do GE.")
    return json.loads(m.group(1))


def _fetch_round_matches(resource_id: str, slug_fase: str, rodada: int) -> List[Dict[str, Any]]:
    api_url = (
        f"https://api.globoesporte.globo.com/tabela/{resource_id}/"
        f"fase/{slug_fase}/rodada/{rodada}/jogos/"
    )
    content = _request_url(api_url)
    raw_matches = json.loads(content)
    matches = []
    for j in raw_matches:
        mandante = j.get("equipes", {}).get("mandante", {})
        visitante = j.get("equipes", {}).get("visitante", {})
        matches.append({
            "id_jogo": j.get("id"),
            "rodada": rodada,
            "data_realizacao": j.get("data_realizacao"),
            "mandante": mandante.get("nome_popular"),
            "mandante_sigla": mandante.get("sigla"),
            "visitante": visitante.get("nome_popular"),
            "visitante_sigla": visitante.get("sigla"),
            "placar_mandante": j.get("placar_oficial_mandante"),
            "placar_visitante": j.get("placar_oficial_visitante"),
            "finalizado": (j.get("placar_oficial_mandante") is not None and j.get("placar_oficial_visitante") is not None)
        })
    return matches


def _is_match_finished(match: Dict[str, Any]) -> bool:
    if "finalizado" in match:
        return bool(match["finalizado"])
    return match.get("placar_mandante") is not None and match.get("placar_visitante") is not None


def _format_rounds_label(
    matches_to_simulate: List[Dict[str, Any]],
    rounds_data: Dict[str, List[Dict[str, Any]]]
) -> str:
    if not matches_to_simulate:
        return "Campeonato encerrado"

    pending_by_round: Dict[int, int] = {}
    for m in matches_to_simulate:
        r = int(m.get("rodada") or 0)
        pending_by_round[r] = pending_by_round.get(r, 0) + 1

    parts: List[str] = []
    range_start = None
    range_end = None

    def _flush_range() -> None:
        nonlocal range_start, range_end
        if range_start is None:
            return
        if range_start == range_end:
            parts.append(f"Rodada {range_start}")
        else:
            parts.append(f"Rodadas {range_start} a {range_end}")
        range_start = None
        range_end = None

    for r in sorted(pending_by_round):
        pending_count = pending_by_round[r]
        total_in_round = len(rounds_data.get(str(r), []))
        if total_in_round > 0 and pending_count < total_in_round:
            _flush_range()
            sufixo = "pendente" if pending_count == 1 else "pendentes"
            parts.append(f"Rodada {r} ({pending_count} {sufixo})")
        else:
            if range_start is None:
                range_start = r
                range_end = r
            elif r == range_end + 1:
                range_end = r
            else:
                _flush_range()
                range_start = r
                range_end = r

    _flush_range()
    return " + ".join(parts)


def _match_key(match: Dict[str, Any]) -> Tuple[Any, ...]:
    if match.get("id_jogo") is not None:
        return ("id", match["id_jogo"])
    return ("tuple", match.get("rodada"), match.get("mandante"), match.get("visitante"))


def select_pending_matches(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Seleciona apenas os confrontos ainda não finalizados a partir dos dados em cache/API,
    ordenados por rodada e sem duplicações (incluindo compatibilidade com caches legados).
    """
    rounds_data = data.get("rounds", {})
    pending_matches: List[Dict[str, Any]] = []
    seen_keys = set()

    for _, round_matches in sorted(rounds_data.items(), key=lambda kv: int(kv[0])):
        for m in round_matches:
            if not _is_match_finished(m):
                key = _match_key(m)
                if key not in seen_keys:
                    seen_keys.add(key)
                    pending_matches.append(m)

    # Compatibilidade com caches antigos que armazenavam jogos adiados em pending_r34
    for m in data.get("pending_r34", []):
        if not _is_match_finished(m):
            key = _match_key(m)
            if key not in seen_keys:
                seen_keys.add(key)
                pending_matches.append(m)

    pending_matches.sort(key=lambda m: int(m.get("rodada") or 0))
    return pending_matches, _format_rounds_label(pending_matches, rounds_data)


def fetch_season_history(
    data: Dict[str, Any],
    force_refresh: bool = False,
    cache_dir: str = ".cache",
) -> List[Dict[str, Any]]:
    """
    Carrega e armazena em cache incremental os jogos finalizados das rodadas 1..rodada_atual.
    Rodadas passadas já 100% finalizadas em cache são reutilizadas sem novas requisições HTTP.
    """
    os.makedirs(cache_dir, exist_ok=True)
    history_path = os.path.join(cache_dir, "ge_2026_history.json")

    resource_id = data.get("resource_id") or DEFAULT_RESOURCE_ID
    slug_fase = data.get("slug_fase") or "fase-unica-campeonato-brasileiro-2026"
    rodada_atual = int(data.get("rodada_atual") or 1)
    existing_rounds = data.get("rounds", {})

    cached_rounds: Dict[str, List[Dict[str, Any]]] = {}
    if not force_refresh and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)
            if (
                cached_payload.get("resource_id") == resource_id
                and cached_payload.get("slug_fase") == slug_fase
            ):
                cached_rounds = cached_payload.get("rounds", {})
        except Exception:
            cached_rounds = {}

    history_rounds: Dict[str, List[Dict[str, Any]]] = {}
    for r in range(1, rodada_atual + 1):
        r_key = str(r)
        if r_key in existing_rounds:
            history_rounds[r_key] = existing_rounds[r_key]
            continue

        cached_r = cached_rounds.get(r_key)
        if not force_refresh and cached_r and all(_is_match_finished(m) for m in cached_r):
            history_rounds[r_key] = cached_r
            continue

        history_rounds[r_key] = _fetch_round_matches(resource_id, slug_fase, r)

    history_rounds = {str(r): history_rounds[str(r)] for r in sorted(int(k) for k in history_rounds)}

    history_payload = {
        "campeonato": data.get("campeonato", "Campeonato Brasileiro 2026"),
        "resource_id": resource_id,
        "slug_fase": slug_fase,
        "rodada_atual": rodada_atual,
        "rounds": history_rounds,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, ensure_ascii=False, indent=2)

    finished_matches: List[Dict[str, Any]] = []
    seen_keys = set()
    for _, round_matches in sorted(history_rounds.items(), key=lambda kv: int(kv[0])):
        for m in round_matches:
            if _is_match_finished(m):
                key = _match_key(m)
                if key not in seen_keys:
                    seen_keys.add(key)
                    finished_matches.append(m)

    finished_matches.sort(
        key=lambda m: (
            int(m.get("rodada") or 0),
            m.get("data_realizacao") or "",
            int(m.get("id_jogo") or 0),
        )
    )
    return finished_matches


def fetch_ge_2026_data(force_refresh: bool = False, cache_dir: str = ".cache") -> Dict[str, Any]:
    """
    Busca os dados oficiais do Brasileirão 2026 da classificação atual e das rodadas restantes.
    Armazena em cache local para evitar tráfego de rede desnecessário.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "ge_2026_data.json")

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            pass  # Recarrega da web se o cache estiver corrompido

    # 1. Baixa a página principal do GE
    html = _request_url(GE_SERIE_A_URL)
    raw_data = _extract_classificacao_payload(html)
    resource_id = _extract_resource_id(html)
    slug_fase = raw_data.get("fase", {}).get("slug", "fase-unica-campeonato-brasileiro-2026")
    rodada_meta = raw_data.get("rodada", {})
    rodada_atual = rodada_meta.get("atual", 1)
    rodada_final = rodada_meta.get("ultima", 38)

    # 2. Processa os 20 clubes da classificação
    standings = []
    for item in raw_data.get("classificacao", []):
        standings.append({
            "id": item.get("equipe_id"),
            "nome": item.get("nome_popular"),
            "sigla": item.get("sigla"),
            "posicao": item.get("ordem"),
            "pontos": int(item.get("pontos", 0)),
            "jogos": int(item.get("jogos", 0)),
            "vitorias": int(item.get("vitorias", 0)),
            "empates": int(item.get("empates", 0)),
            "derrotas": int(item.get("derrotas", 0)),
            "gols_pro": int(item.get("gols_pro", 0)),
            "gols_contra": int(item.get("gols_contra", 0)),
            "saldo_gols": int(item.get("saldo_gols", 0)),
            "aproveitamento": float(item.get("aproveitamento", 0.0)),
            "ultimos_jogos": item.get("ultimos_jogos", [])
        })

    # Ordena por critérios oficiais da CBF
    standings.sort(key=lambda t: (-t["pontos"], -t["vitorias"], -t["saldo_gols"], -t["gols_pro"]))
    for i, t in enumerate(standings, start=1):
        t["posicao"] = i

    # 3. Busca os jogos da rodada atual até a rodada final
    rounds_data: Dict[str, List[Dict[str, Any]]] = {}
    for r in range(rodada_atual, rodada_final + 1):
        rounds_data[str(r)] = _fetch_round_matches(resource_id, slug_fase, r)

    # 4. Detecta automaticamente jogos adiados de rodadas anteriores (< rodada_atual)
    pending_by_team: Dict[str, int] = {}
    for round_matches in rounds_data.values():
        for m in round_matches:
            if not _is_match_finished(m):
                pending_by_team[m["mandante"]] = pending_by_team.get(m["mandante"], 0) + 1
                pending_by_team[m["visitante"]] = pending_by_team.get(m["visitante"], 0) + 1

    def _all_teams_complete() -> bool:
        return all(t["jogos"] + pending_by_team.get(t["nome"], 0) >= rodada_final for t in standings)

    if not _all_teams_complete():
        for r in range(rodada_atual - 1, 0, -1):
            r_matches = _fetch_round_matches(resource_id, slug_fase, r)
            pending_in_r = [m for m in r_matches if not _is_match_finished(m)]
            if pending_in_r:
                rounds_data[str(r)] = r_matches
                for m in pending_in_r:
                    pending_by_team[m["mandante"]] = pending_by_team.get(m["mandante"], 0) + 1
                    pending_by_team[m["visitante"]] = pending_by_team.get(m["visitante"], 0) + 1
            if _all_teams_complete():
                break

    rounds_data = {str(r): rounds_data[str(r)] for r in sorted(int(k) for k in rounds_data)}

    payload = {
        "campeonato": "Campeonato Brasileiro 2026",
        "edicao": raw_data.get("edicao", {}).get("nome", "Campeonato Brasileiro 2026"),
        "rodada_atual": rodada_atual,
        "rodada_final": rodada_final,
        "resource_id": resource_id,
        "slug_fase": slug_fase,
        "standings": standings,
        "rounds": rounds_data,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    data = fetch_ge_2026_data(force_refresh=True)
    pending_matches, rodadas_label = select_pending_matches(data)
    print(f"Carregado: {data['campeonato']} (Rodada atual: {data['rodada_atual']})")
    print(f"Total de clubes: {len(data['standings'])}")
    print(f"Total de jogos pendentes: {len(pending_matches)} ({rodadas_label})")
    for r, matches in data["rounds"].items():
        pendentes = sum(1 for m in matches if not _is_match_finished(m))
        print(f"Rodada {r}: {len(matches)} jogos ({pendentes} pendentes)")
