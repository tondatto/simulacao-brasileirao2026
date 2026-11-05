"""
Módulo estatístico inspirado na metodologia do Prof. Rogemar Riffel (IF-UFRGS):
- Peso Composto 50/50: 50% histórico da temporada + 50% forma recente (últimas 5 rodadas)
- Ajuste de vantagem do mandante (fator casa)
- Probabilidade base de empate calibrada para o Brasileirão (~26%)
- Fator de 'zebra' de 5% (inversão estocástica de probabilidades)
"""

from typing import Dict, Any, List, Tuple
import math


def calculate_team_strengths(standings: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calcula a força de cada equipe baseada no peso composto 50/50:
    50% aproveitamento histórico da temporada + 50% aproveitamento nas últimas 5 rodadas.
    """
    strengths = {}

    for t in standings:
        nome = t["nome"]
        jogos = t["jogos"]
        pontos = t["pontos"]

        # Aproveitamento histórico (0.0 a 1.0)
        if jogos > 0:
            aprov_hist = pontos / (3 * jogos)
        else:
            aprov_hist = 0.5

        # Aproveitamento recente (últimas 5 rodadas)
        ultimos = t.get("ultimos_jogos", [])
        if ultimos:
            pts_recentes = 0
            for r in ultimos:
                if str(r).lower() == "v":
                    pts_recentes += 3
                elif str(r).lower() == "e":
                    pts_recentes += 1
            max_pts = 3 * len(ultimos)
            aprov_rec = pts_recentes / max_pts if max_pts > 0 else aprov_hist
        else:
            aprov_rec = aprov_hist

        # Força composta 50/50
        forca_composta = 0.5 * aprov_hist + 0.5 * aprov_rec

        strengths[nome] = {
            "aprov_hist": aprov_hist,
            "aprov_rec": aprov_rec,
            "forca_composta": forca_composta,
            "pontos": pontos,
            "vitorias": t["vitorias"],
            "saldo_gols": t["saldo_gols"],
            "gols_pro": t["gols_pro"]
        }

    return strengths


def calculate_match_probabilities(
    mandante_nome: str,
    visitante_nome: str,
    strengths: Dict[str, Dict[str, float]],
    fator_casa: float = 0.12,
    taxa_empate_base: float = 0.26,
    fator_zebra: float = 0.05
) -> Tuple[float, float, float]:
    """
    Calcula as probabilidades de [Vitória Mandante, Empate, Vitória Visitante]
    para um confronto específico, aplicando o modelo Riffel e o fator de zebra de 5%.
    """
    s_mandante = strengths.get(mandante_nome, {}).get("forca_composta", 0.5)
    s_visitante = strengths.get(visitante_nome, {}).get("forca_composta", 0.5)

    # Força com vantagem de jogar em casa
    f_mandante = max(0.05, s_mandante + fator_casa)
    f_visitante = max(0.05, s_visitante)

    # Diferença de forças
    diff = f_mandante - f_visitante

    # Probabilidade de empate diminui suavemente conforme o desequilíbrio entre os times aumenta
    p_empate = taxa_empate_base * math.exp(-(diff ** 2) / (2 * (0.25 ** 2)))
    p_empate = max(0.15, min(0.35, p_empate))

    p_decisao = 1.0 - p_empate

    # Divisão entre mandante e visitante proporcional às forças relativas
    total_forca = f_mandante + f_visitante
    p_mandante = p_decisao * (f_mandante / total_forca)
    p_visitante = p_decisao * (f_visitante / total_forca)

    # Aplicação do Fator de Zebra (inversão estocástica de 5%)
    # Com 5% de probabilidade, o resultado favorece o time não favorito
    if fator_zebra > 0:
        p_m_zebra = (1.0 - fator_zebra) * p_mandante + fator_zebra * p_visitante
        p_v_zebra = (1.0 - fator_zebra) * p_visitante + fator_zebra * p_mandante
        p_mandante, p_visitante = p_m_zebra, p_v_zebra

    # Normalização para garantir que somem exatamente 1.0
    soma = p_mandante + p_empate + p_visitante
    return p_mandante / soma, p_empate / soma, p_visitante / soma


def prepare_matches_with_probabilities(
    matches: List[Dict[str, Any]],
    strengths: Dict[str, Dict[str, float]],
    fator_zebra: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Adiciona as probabilidades calculadas a cada confronto da lista.
    """
    enriched_matches = []
    for m in matches:
        mandante = m["mandante"]
        visitante = m["visitante"]
        pm, pe, pv = calculate_match_probabilities(
            mandante, visitante, strengths, fator_zebra=fator_zebra
        )
        enriched_matches.append({
            **m,
            "p_mandante": pm,
            "p_empate": pe,
            "p_visitante": pv
        })
    return enriched_matches


if __name__ == "__main__":
    from src.ge_fetcher import fetch_ge_2026_data

    data = fetch_ge_2026_data()
    strengths = calculate_team_strengths(data["standings"])

    print("--- Força Composta (50% Histórico + 50% Recente) ---")
    for t in data["standings"][:5]:
        nome = t["nome"]
        st = strengths[nome]
        print(f"{nome:15s}: Hist={st['aprov_hist']*100:4.1f}% | Rec={st['aprov_rec']*100:4.1f}% | Composta={st['forca_composta']*100:4.1f}%")

    print("\n--- Exemplo: Palmeiras x Flamengo (Rodada 36) ---")
    pm, pe, pv = calculate_match_probabilities("Palmeiras", "Flamengo", strengths)
    print(f"Palmeiras: {pm*100:.1f}% | Empate: {pe*100:.1f}% | Flamengo: {pv*100:.1f}%")
