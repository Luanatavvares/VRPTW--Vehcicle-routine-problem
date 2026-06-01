"""
=============================================================================
CVRP Instance Generator — Capacitated Vehicle Routing Problem
=============================================================================
Gera instâncias no formato Solomon-like (.txt) de forma interativa,
pedindo ao usuário os parâmetros desejados.

Uso:
    python cvrp_generator.py                  (modo interativo)
    python cvrp_generator.py --clientes 5 --veiculos 2 --capacidade 30
    python cvrp_generator.py --ajuda

Compatível com Python 3.8+. Sem dependências externas.
=============================================================================
"""

import math
import random
import os
import sys
import argparse
from itertools import permutations


# =============================================================================
# BANNER
# =============================================================================

BANNER = r"""
  ██████╗██╗   ██╗██████╗ ██████╗
 ██╔════╝██║   ██║██╔══██╗██╔══██╗
 ██║     ██║   ██║██████╔╝██████╔╝
 ██║     ╚██╗ ██╔╝██╔══██╗██╔═══╝
 ╚██████╗ ╚████╔╝ ██║  ██║██║
  ╚═════╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  Gerador de Instâncias
"""

SEP  = "=" * 62
SEP2 = "-" * 62


# =============================================================================
# GEOMETRIA
# =============================================================================

def euclidean(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(depot: tuple, customers: list) -> dict:
    nodes = {0: depot}
    for c in customers:
        nodes[c["id"]] = (c["x"], c["y"])
    dist = {}
    for i in nodes:
        for j in nodes:
            dist[(i, j)] = euclidean(nodes[i], nodes[j])
    return dist


# =============================================================================
# GERAÇÃO DE COORDENADAS (inspirado em Solomon)
# =============================================================================

def gerar_deposito(grid: int) -> tuple:
    """Depósito próximo ao centro do plano."""
    cx, cy = grid // 2, grid // 2
    offset = grid // 10
    x = cx + random.randint(-offset, offset)
    y = cy + random.randint(-offset, offset)
    return (x, y)


def gerar_clientes(n: int, grid: int, depot: tuple,
                   dem_min: int, dem_max: int, seed: int) -> list:
    """
    Gera n clientes com coordenadas inteiras únicas e afastadas do depósito.
    Estratégia inspirada em Solomon:
      - distribui clientes em setores angulares para garantir espalhamento;
      - distância mínima ao depósito = grid // 6;
      - sem coordenadas repetidas.
    """
    random.seed(seed)
    customers = []
    ocupados = {depot}
    min_dist_depot = max(grid // 6, 3)
    margin = max(2, grid // 20)

    tentativas = 0
    max_tentativas = 10_000

    # Divide o círculo em setores para distribuir os clientes
    angulos = [2 * math.pi * i / n for i in range(n)]
    random.shuffle(angulos)

    for i, angulo in enumerate(angulos):
        while tentativas < max_tentativas:
            tentativas += 1
            # Raio aleatório com distância mínima garantida
            r = random.randint(min_dist_depot, grid // 2 - margin)
            # Variação angular dentro do setor
            variacao = (2 * math.pi / n) * 0.4
            ang = angulo + random.uniform(-variacao, variacao)
            cx = depot[0] + int(r * math.cos(ang))
            cy = depot[1] + int(r * math.sin(ang))

            # Mantém dentro do grid
            cx = max(margin, min(grid - margin, cx))
            cy = max(margin, min(grid - margin, cy))
            pos = (cx, cy)

            # Garante unicidade e distância mínima ao depósito
            if pos not in ocupados and euclidean(pos, depot) >= min_dist_depot:
                ocupados.add(pos)
                dem = random.randint(dem_min, dem_max)
                customers.append({
                    "id": i + 1,
                    "x": cx,
                    "y": cy,
                    "demand": dem
                })
                break
        else:
            raise RuntimeError(
                f"Não foi possível posicionar o cliente {i+1} após "
                f"{max_tentativas} tentativas. Tente um grid maior."
            )

    return customers


# =============================================================================
# CÁLCULO DE CAPACIDADE VIÁVEL
# =============================================================================

def capacidade_sugerida(customers: list, num_vehicles: int,
                         fator: float = 1.5) -> int:
    """
    Sugere capacidade mínima viável:
      - Garante que a demanda máxima individual caiba num veículo;
      - Garante que a demanda total possa ser distribuída entre os veículos;
      - Adiciona folga (fator) para que existam múltiplas soluções viáveis.
    """
    demandas = [c["demand"] for c in customers]
    dem_total = sum(demandas)
    dem_max   = max(demandas)

    # Capacidade mínima absoluta (maior demanda individual)
    cap_min = dem_max

    # Capacidade para distribuir a carga média com folga
    cap_media = math.ceil((dem_total / num_vehicles) * fator)

    return max(cap_min, cap_media)


# =============================================================================
# SOLVER FORÇA BRUTA (para solução ótima embutida)
# =============================================================================

def resolver_bruta(depot: tuple, customers: list,
                   num_vehicles: int, capacity: int) -> dict:
    """Resolve o CVRP por força bruta e retorna a solução ótima."""
    dist       = build_distance_matrix(depot, customers)
    demand_map = {c["id"]: c["demand"] for c in customers}
    cust_ids   = [c["id"] for c in customers]

    def route_cost(route):
        if not route:
            return 0.0
        total = dist[(0, route[0])]
        for i in range(len(route) - 1):
            total += dist[(route[i], route[i + 1])]
        total += dist[(route[-1], 0)]
        return total

    def assignments(ids, k):
        if not ids:
            yield [[] for _ in range(k)]
            return
        first, rest = ids[0], ids[1:]
        for sub in assignments(rest, k):
            for v in range(k):
                new = [list(r) for r in sub]
                new[v] = [first] + new[v]
                yield new

    best_cost   = math.inf
    best_routes = None

    for partition in assignments(cust_ids, num_vehicles):
        # Viabilidade de capacidade
        if any(sum(demand_map[c] for c in g) > capacity for g in partition):
            continue

        group_orders = []
        total_cost   = 0.0

        for group in partition:
            if not group:
                group_orders.append([])
                continue
            local_best = min(permutations(group),
                             key=lambda p: route_cost(list(p)))
            local_best = list(local_best)
            group_orders.append(local_best)
            total_cost += route_cost(local_best)

        if total_cost < best_cost:
            best_cost   = total_cost
            best_routes = group_orders

    return {"cost": round(best_cost, 4), "routes": best_routes}


# =============================================================================
# FORMATAÇÃO DO ARQUIVO
# =============================================================================

def formatar_instancia(nome: str, num_vehicles: int, capacity: int,
                       depot: tuple, customers: list,
                       solucao: dict) -> str:
    linhas = []
    linhas.append(nome)
    linhas.append("")
    linhas.append("VEHICLE")
    linhas.append(f"{'NUMBER':<10} {'CAPACITY'}")
    linhas.append(f"{num_vehicles:<10} {capacity}")
    linhas.append("")
    linhas.append("CUSTOMER")
    linhas.append(f"{'CUST NO.':<10} {'XCOORD.':<10} {'YCOORD.':<10} {'DEMAND'}")
    linhas.append(f"{'0':<10} {depot[0]:<10} {depot[1]:<10} {'0'}")
    for c in customers:
        linhas.append(f"{c['id']:<10} {c['x']:<10} {c['y']:<10} {c['demand']}")

    linhas.append("")
    linhas.append("OPTIMAL SOLUTION")
    demand_map = {c["id"]: c["demand"] for c in customers}
    dist       = build_distance_matrix(depot, customers)

    for i, route in enumerate(solucao["routes"]):
        if route:
            path = " -> ".join(str(n) for n in [0] + route + [0])
            dem  = sum(demand_map[c] for c in route)

            def rc(r):
                t = dist[(0, r[0])]
                for j in range(len(r) - 1):
                    t += dist[(r[j], r[j+1])]
                t += dist[(r[-1], 0)]
                return round(t, 4)

            custo = rc(route)
            linhas.append(
                f"Route {i+1}: {path}  "
                f"[demand={dem}, cost={custo}]"
            )
        else:
            linhas.append(f"Route {i+1}: idle")

    linhas.append(
        f"Total optimal cost (Euclidean): {solucao['cost']}"
    )

    return "\n".join(linhas)


# =============================================================================
# INTERFACE INTERATIVA
# =============================================================================

def perguntar_int(prompt: str, minimo: int, maximo: int,
                  padrao: int = None) -> int:
    """Lê um inteiro do usuário com validação."""
    while True:
        sufixo = f" [{padrao}]" if padrao is not None else ""
        entrada = input(f"  {prompt}{sufixo}: ").strip()
        if entrada == "" and padrao is not None:
            return padrao
        try:
            valor = int(entrada)
            if minimo <= valor <= maximo:
                return valor
            print(f"  ⚠  Digite um valor entre {minimo} e {maximo}.")
        except ValueError:
            print("  ⚠  Entrada inválida. Digite um número inteiro.")


def perguntar_sim_nao(prompt: str, padrao: bool = True) -> bool:
    suf = "[S/n]" if padrao else "[s/N]"
    while True:
        entrada = input(f"  {prompt} {suf}: ").strip().lower()
        if entrada == "":
            return padrao
        if entrada in ("s", "sim", "y", "yes"):
            return True
        if entrada in ("n", "nao", "não", "no"):
            return False
        print("  ⚠  Digite S (sim) ou N (não).")


def modo_interativo() -> dict:
    """Conduz o usuário pelas perguntas e retorna os parâmetros."""
    print(BANNER)
    print(SEP)
    print("  Bem-vindo ao Gerador de Instâncias CVRP")
    print("  Responda as perguntas abaixo (Enter = valor padrão)")
    print(SEP)

    print("\n── Identificação da instância ──────────────────────────")
    nome = input("  Nome da instância [I_auto]: ").strip()
    if not nome:
        nome = "I_auto"

    print("\n── Estrutura da instância ──────────────────────────────")
    num_vehicles = perguntar_int("Número de veículos", 1, 20, padrao=2)
    num_clients  = perguntar_int("Número de clientes", 1, 12, padrao=3)

    print("\n── Espaço de coordenadas ───────────────────────────────")
    grid = perguntar_int(
        "Tamanho do grid (coordenadas de 0 a N)", 20, 200, padrao=50
    )

    print("\n── Demandas dos clientes ────────────────────────────────")
    dem_min = perguntar_int("Demanda mínima por cliente", 1, 50, padrao=3)
    dem_max = perguntar_int(
        f"Demanda máxima por cliente (>= {dem_min})", dem_min, 100, padrao=max(dem_min, 8)
    )

    print("\n── Capacidade dos veículos ─────────────────────────────")
    print("  (deixe em branco para calcular automaticamente)")
    cap_input = input("  Capacidade dos veículos [auto]: ").strip()

    print("\n── Reprodutibilidade ───────────────────────────────────")
    seed = perguntar_int("Semente aleatória (seed)", 0, 999_999, padrao=42)

    print("\n── Arquivo de saída ────────────────────────────────────")
    arquivo = input(f"  Nome do arquivo [{nome}.txt]: ").strip()
    if not arquivo:
        arquivo = f"{nome}.txt"
    if not arquivo.endswith(".txt"):
        arquivo += ".txt"

    return {
        "nome":         nome,
        "num_vehicles": num_vehicles,
        "num_clients":  num_clients,
        "grid":         grid,
        "dem_min":      dem_min,
        "dem_max":      dem_max,
        "cap_input":    cap_input,
        "seed":         seed,
        "arquivo":      arquivo,
    }


# =============================================================================
# GERAÇÃO COMPLETA
# =============================================================================

def gerar(params: dict, verbose: bool = True) -> str:
    """
    Recebe os parâmetros, gera a instância, resolve e salva em arquivo.
    Retorna o caminho do arquivo gerado.
    """
    nome         = params["nome"]
    num_vehicles = params["num_vehicles"]
    num_clients  = params["num_clients"]
    grid         = params["grid"]
    dem_min      = params["dem_min"]
    dem_max      = params["dem_max"]
    cap_input    = params.get("cap_input", "").strip()
    seed         = params.get("seed", 42)
    arquivo      = params.get("arquivo", f"{nome}.txt")

    if verbose:
        print(f"\n{SEP}")
        print(f"  Gerando instância: {nome}")
        print(SEP2)

    # 1. Depósito
    random.seed(seed)
    depot = gerar_deposito(grid)

    # 2. Clientes
    customers = gerar_clientes(
        num_clients, grid, depot, dem_min, dem_max, seed
    )

    # 3. Capacidade
    if cap_input and cap_input.lstrip("-").isdigit():
        capacity = int(cap_input)
        dem_max_c = max(c["demand"] for c in customers)
        if capacity < dem_max_c:
            print(f"\n  ⚠  Capacidade {capacity} < demanda máxima "
                  f"individual {dem_max_c}. Ajustando para {dem_max_c}.")
            capacity = dem_max_c
    else:
        capacity = capacidade_sugerida(customers, num_vehicles)

    # 4. Resolver (força bruta — só viável para instâncias pequenas)
    limite_bruta = 9  # clientes
    if num_clients <= limite_bruta:
        if verbose:
            print(f"  Resolvendo por força bruta ({num_clients} clientes)...")
        solucao = resolver_bruta(depot, customers, num_vehicles, capacity)
    else:
        if verbose:
            print(
                f"  ⚠  {num_clients} clientes > limite ({limite_bruta}) "
                f"para força bruta.\n"
                f"     Solução ótima não calculada — instância salva sem ela."
            )
        solucao = {"cost": None, "routes": [[] for _ in range(num_vehicles)]}

    # 5. Formatar e salvar
    conteudo = formatar_instancia(
        nome, num_vehicles, capacity, depot, customers, solucao
    )

    with open(arquivo, "w") as f:
        f.write(conteudo)

    # 6. Exibir resumo
    if verbose:
        dem_total = sum(c["demand"] for c in customers)
        print(f"\n{SEP}")
        print(f"  Instância gerada com sucesso!")
        print(SEP2)
        print(f"  {'Nome':<28}: {nome}")
        print(f"  {'Arquivo':<28}: {arquivo}")
        print(f"  {'Veículos':<28}: {num_vehicles}")
        print(f"  {'Clientes':<28}: {num_clients}")
        print(f"  {'Capacidade por veículo':<28}: {capacity}")
        print(f"  {'Demanda total':<28}: {dem_total}")
        print(f"  {'Depósito':<28}: {depot}")
        print(f"  {'Grid':<28}: 0–{grid} × 0–{grid}")
        print(f"  {'Seed':<28}: {seed}")

        if solucao["cost"] is not None:
            print(SEP2)
            print(f"  Solução ótima (força bruta):")
            demand_map = {c["id"]: c["demand"] for c in customers}
            dist = build_distance_matrix(depot, customers)
            for i, route in enumerate(solucao["routes"]):
                if route:
                    path = " → ".join(str(n) for n in [0] + route + [0])
                    dem  = sum(demand_map[c] for c in route)
                    print(f"    V{i+1}: {path}  (dem={dem})")
                else:
                    print(f"    V{i+1}: ocioso")
            print(f"  Custo ótimo total: {solucao['cost']:.4f}")

        print(SEP)

    return arquivo


# =============================================================================
# MODO LOTE: gera múltiplas instâncias de uma vez
# =============================================================================

def modo_lote():
    """Permite gerar várias instâncias em sequência."""
    print(BANNER)
    print(SEP)
    print("  Modo lote — geração de múltiplas instâncias")
    print(SEP)

    continuar = True
    gerados   = []

    while continuar:
        params  = modo_interativo()
        arquivo = gerar(params, verbose=True)
        gerados.append(arquivo)

        print()
        continuar = perguntar_sim_nao("Gerar outra instância?", padrao=False)

    print(f"\n{SEP}")
    print(f"  {len(gerados)} instância(s) gerada(s):")
    for f in gerados:
        print(f"    • {f}")
    print(SEP)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gerador de instâncias CVRP no formato Solomon-like.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--clientes",   type=int, help="Número de clientes (1–12)")
    parser.add_argument("--veiculos",   type=int, help="Número de veículos (1–20)")
    parser.add_argument("--capacidade", type=int, help="Capacidade por veículo")
    parser.add_argument("--grid",       type=int, default=50,
                        help="Tamanho do espaço de coordenadas (padrão: 50)")
    parser.add_argument("--dem-min",    type=int, default=3,
                        help="Demanda mínima por cliente (padrão: 3)")
    parser.add_argument("--dem-max",    type=int, default=8,
                        help="Demanda máxima por cliente (padrão: 8)")
    parser.add_argument("--nome",       type=str, default="I_auto",
                        help="Nome da instância (padrão: I_auto)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Semente aleatória (padrão: 42)")
    parser.add_argument("--saida",      type=str,
                        help="Nome do arquivo de saída (padrão: <nome>.txt)")
    parser.add_argument("--lote",       action="store_true",
                        help="Modo lote: gera múltiplas instâncias em sequência")

    args = parser.parse_args()

    # Modo lote
    if args.lote:
        modo_lote()
        return

    # Modo linha de comando (todos os parâmetros fornecidos)
    if args.clientes and args.veiculos:
        params = {
            "nome":         args.nome,
            "num_vehicles": args.veiculos,
            "num_clients":  args.clientes,
            "grid":         args.grid,
            "dem_min":      args.dem_min,
            "dem_max":      args.dem_max,
            "cap_input":    str(args.capacidade) if args.capacidade else "",
            "seed":         args.seed,
            "arquivo":      args.saida or f"{args.nome}.txt",
        }
        gerar(params, verbose=True)
        return

    # Modo interativo (padrão)
    try:
        params  = modo_interativo()
        gerar(params, verbose=True)
    except KeyboardInterrupt:
        print("\n\n  Operação cancelada pelo usuário.")
        sys.exit(0)


if __name__ == "__main__":
    main()
