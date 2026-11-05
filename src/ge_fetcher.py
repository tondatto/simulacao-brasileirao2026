"""
Módulo para extração de dados do Campeonato Brasileiro 2026 diretamente do Globo Esporte (GE),
baseado na arquitetura do repositório ezefranca/campeonato-brasileiro-api.
Inclui cache local em disco em .cache/ para execução offline e máxima performance.
"""

import json
import os
import re
import urllib.request
from typing import Dict, Any, List, Optional

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
    rodada_atual = rodada_meta.get("atual", 34)
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

    # 3. Busca os jogos das rodadas restantes (35 a 38)
    rounds_data = {}
    for r in range(35, rodada_final + 1):
        rounds_data[str(r)] = _fetch_round_matches(resource_id, slug_fase, r)

    # 4. Verifica jogos pendentes da rodada 34
    pending_r34 = []
    r34_matches = _fetch_round_matches(resource_id, slug_fase, 34)
    for m in r34_matches:
        if not m["finalizado"]:
            pending_r34.append(m)

    payload = {
        "campeonato": "Campeonato Brasileiro 2026",
        "edicao": raw_data.get("edicao", {}).get("nome", "Campeonato Brasileiro 2026"),
        "rodada_atual": rodada_atual,
        "rodada_final": rodada_final,
        "resource_id": resource_id,
        "slug_fase": slug_fase,
        "standings": standings,
        "rounds": rounds_data,
        "pending_r34": pending_r34
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    data = fetch_ge_2026_data(force_refresh=True)
    print(f"Carregado: {data['campeonato']}")
    print(f"Total de clubes: {len(data['standings'])}")
    print(f"Jogos pendentes rodada 34: {len(data['pending_r34'])}")
    for r, matches in data["rounds"].items():
        print(f"Rodada {r}: {len(matches)} jogos")
