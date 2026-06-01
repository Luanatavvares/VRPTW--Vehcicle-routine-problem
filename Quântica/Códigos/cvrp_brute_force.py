"""
=============================================================================
CVRP Brute Force Solver — Capacitated Vehicle Routing Problem
=============================================================================
Lê instâncias no formato Solomon-like (.txt) e resolve por enumeração
completa (força bruta), garantindo a solução ótima exata.

Uso:
    python cvrp_brute_force.py instancia.txt
    python cvrp_brute_force.py instancia.txt --verbose
    python cvrp_brute_force.py --all         (resolve todos os .txt no diretório)

Compatível com Python 3.8+. Sem dependências externas.
=============================================================================
"""

import math
import sys
import os
import argparse
import time
from itertools import permutations


# =============================================================================
# LEITURA DE INSTÂNCIA
# =============================================================================

def parse_instance(filepath: str) -> dict:
    """
    Lê um arquivo .txt no formato Solomon-like e retorna um dicionário com:
      - name        : nome da instância
      - num_vehicles: número de veículos
      - capacity    : capacidade por veículo
      - depot       : (x, y) do depósito
      - customers   : lista de dicts {id, x, y, demand}
    """
    instance = {}
    customers = []

    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    # Nome: primeira linha não vazia
    instance["name"] = next(l for l in lines if l)

    # Veículos
    reading_vehicles = False
    reading_customers = False

    for i, line in enumerate(lines):
        if not line:
            continue

        if line.startswith("VEHICLE"):
            reading_vehicles = True
            reading_customers = False
            continue

        if line.startswith("CUSTOMER"):
            reading_vehicles = False
            reading_customers = True
            continue

        if line.startswith("OPTIMAL"):
            reading_customers = False
            continue

        # Linha de cabeçalho da seção veículos (NUMBER CAPACITY)
        if reading_vehicles and line.startswith("NUMBER"):
            continue

        # Dados dos veículos
        if reading_vehicles:
            parts = line.split()
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                instance["num_vehicles"] = int(parts[0])
                instance["capacity"] = int(parts[1])
                reading_vehicles = False
            continue

        # Cabeçalho da seção clientes
        if reading_customers and line.startswith("CUST"):
            continue

        # Dados dos clientes
        if reading_customers:
            parts = line.split()
            if len(parts) >= 4:
                cid = int(parts[0])
                x   = int(parts[1])
                y   = int(parts[2])
                dem = int(parts[3])
                if cid == 0:
                    instance["depot"] = (x, y)
                    instance["depot_id"] = cid
                else:
                    customers.append({"id": cid, "x": x, "y": y, "demand": dem})

    instance["customers"] = customers
    return instance


# =============================================================================
# GEOMETRIA
# =============================================================================

def euclidean(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(instance: dict) -> dict:
    """Retorna dict dist[(i,j)] para todos os nós (0 = depósito)."""
    nodes = {0: instance["depot"]}
    for c in instance["customers"]:
        nodes[c["id"]] = (c["x"], c["y"])

    dist = {}
    for i in nodes:
        for j in nodes:
            dist[(i, j)] = euclidean(nodes[i], nodes[j])
    return dist


# =============================================================================
# PARTICIONAMENTO EM ROTAS
# =============================================================================

def partition_customers(customer_ids: list, num_vehicles: int):
    """
    Gera todas as formas de distribuir `customer_ids` entre `num_vehicles`
    veículos (alguns podem ficar vazios). Usa abordagem de número de Stirling /
    partições ordenadas.

    Para n clientes e k veículos, gera todas as k-tuplas de subconjuntos
    disjuntos que cobrem todos os clientes.
    """
    n = len(customer_ids)
    k = num_vehicles

    # Atribuição de cada cliente a um veículo: k^n combinações
    # Ex.: [0,1,0] = cliente 0 no veículo 0, cliente 1 no veículo 1, etc.
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

    yield from assignments(customer_ids, k)


def route_cost(route: list, dist: dict, depot_id: int = 0) -> float:
    """Custo de uma rota simples: depot -> c1 -> c2 -> ... -> depot."""
    if not route:
        return 0.0
    total = dist[(depot_id, route[0])]
    for i in range(len(route) - 1):
        total += dist[(route[i], route[i + 1])]
    total += dist[(route[-1], depot_id)]
    return total


def route_demand(route: list, demand_map: dict) -> int:
    return sum(demand_map[c] for c in route)


# =============================================================================
# SOLVER FORÇA BRUTA
# =============================================================================

def solve_brute_force(instance: dict, verbose: bool = False) -> dict:
    """
    Resolve o CVRP por força bruta completa:
      1. Gera todas as partições dos clientes nos veículos.
      2. Para cada partição, gera todas as permutações de cada rota.
      3. Verifica viabilidade (capacidade).
      4. Calcula custo e mantém o mínimo.

    Retorna dicionário com a solução ótima.
    """
    t0 = time.time()

    num_vehicles = instance["num_vehicles"]
    capacity     = instance["capacity"]
    customers    = instance["customers"]
    depot_id     = 0

    dist       = build_distance_matrix(instance)
    demand_map = {c["id"]: c["demand"] for c in customers}
    cust_ids   = [c["id"] for c in customers]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Instância : {instance['name']}")
        print(f"  Veículos  : {num_vehicles}  |  Capacidade : {capacity}")
        print(f"  Clientes  : {len(cust_ids)}  →  {cust_ids}")
        print(f"  Depósito  : {instance['depot']}")
        print(f"{'='*60}")

    best_cost   = math.inf
    best_routes = None
    feasible_count = 0
    total_evaluated = 0

    # -------------------------------------------------------------------------
    # Iteração principal
    # -------------------------------------------------------------------------
    for partition in partition_customers(cust_ids, num_vehicles):
        # Verifica capacidade de cada grupo antes de permutar
        feasible_partition = True
        for group in partition:
            if route_demand(group, demand_map) > capacity:
                feasible_partition = False
                break
        if not feasible_partition:
            continue

        # Permuta cada grupo para encontrar melhor ordem de visita
        # Grupos vazios têm custo 0 e permutação trivial
        group_best_costs  = []
        group_best_orders = []

        for group in partition:
            if not group:
                group_best_costs.append(0.0)
                group_best_orders.append([])
                continue

            local_best_cost  = math.inf
            local_best_order = None

            for perm in permutations(group):
                perm = list(perm)
                c = route_cost(perm, dist, depot_id)
                total_evaluated += 1
                if c < local_best_cost:
                    local_best_cost  = c
                    local_best_order = perm

            group_best_costs.append(local_best_cost)
            group_best_orders.append(local_best_order)

        total_cost = sum(group_best_costs)
        feasible_count += 1

        if total_cost < best_cost:
            best_cost   = total_cost
            best_routes = group_best_orders

    elapsed = time.time() - t0

    # -------------------------------------------------------------------------
    # Formata resultado
    # -------------------------------------------------------------------------
    if best_routes is None:
        return {"feasible": False, "instance": instance["name"]}

    routes_fmt = []
    for i, route in enumerate(best_routes):
        if route:
            path = " → ".join(str(n) for n in [depot_id] + route + [depot_id])
            dem  = route_demand(route, demand_map)
            cost = route_cost(route, dist, depot_id)
            routes_fmt.append({
                "vehicle": i + 1,
                "path": path,
                "demand": dem,
                "cost": round(cost, 4),
            })
        else:
            routes_fmt.append({
                "vehicle": i + 1,
                "path": "ocioso",
                "demand": 0,
                "cost": 0.0,
            })

    return {
        "feasible":        True,
        "instance":        instance["name"],
        "optimal_cost":    round(best_cost, 4),
        "routes":          routes_fmt,
        "feasible_solutions_checked": feasible_count,
        "routes_evaluated": total_evaluated,
        "elapsed_sec":     round(elapsed, 4),
    }


# =============================================================================
# IMPRESSÃO DE RESULTADOS
# =============================================================================

def print_solution(sol: dict, instance: dict):
    SEP  = "=" * 60
    SEP2 = "-" * 60

    if not sol["feasible"]:
        print(f"\n{SEP}")
        print(f"  INSTÂNCIA: {sol['instance']}")
        print(f"  ❌  Nenhuma solução viável encontrada.")
        print(SEP)
        return

    print(f"\n{SEP}")
    print(f"  INSTÂNCIA : {sol['instance']}")
    print(f"  Veículos  : {instance['num_vehicles']}  |  "
          f"Capacidade : {instance['capacity']}")
    print(SEP2)
    print(f"  {'VEÍ':>3}  {'ROTA':<35}  {'DEMANDA':>7}  {'CUSTO':>10}")
    print(SEP2)

    for r in sol["routes"]:
        status = "" if r["path"] == "ocioso" else ""
        print(f"  {r['vehicle']:>3}  {r['path']:<35}  "
              f"{r['demand']:>7}  {r['cost']:>10.4f}  {status}")

    print(SEP2)
    print(f"  {'CUSTO ÓTIMO TOTAL':>40} : {sol['optimal_cost']:.4f}")
    print(SEP2)
    print(f"  Soluções viáveis avaliadas : {sol['feasible_solutions_checked']}")
    print(f"  Rotas permutadas avaliadas : {sol['routes_evaluated']}")
    print(f"  Tempo de execução          : {sol['elapsed_sec']} s")
    print(SEP)


# =============================================================================
# VALIDAÇÃO DA SOLUÇÃO
# =============================================================================

def validate_solution(sol: dict, instance: dict) -> bool:
    """Verifica se a solução cobre todos os clientes e respeita capacidades."""
    if not sol["feasible"]:
        return False

    capacity   = instance["capacity"]
    all_cust   = {c["id"] for c in instance["customers"]}
    visited    = set()
    valid      = True
    errors     = []

    for r in sol["routes"]:
        if r["path"] == "ocioso":
            continue
        # Extrai IDs da string "0 → 1 → 2 → 0"
        nodes = [int(n.strip()) for n in r["path"].split("→")]
        route_nodes = nodes[1:-1]  # remove depot das pontas

        for c in route_nodes:
            if c in visited:
                errors.append(f"  ⚠ Cliente {c} visitado mais de uma vez!")
                valid = False
            visited.add(c)

        if r["demand"] > capacity:
            errors.append(f"  ⚠ Veículo {r['vehicle']}: demanda {r['demand']} "
                          f"> capacidade {capacity}!")
            valid = False

    missing = all_cust - visited
    if missing:
        errors.append(f"  ⚠ Clientes não atendidos: {missing}")
        valid = False

    if valid:
        print(f"  ✅  Validação OK — todos os {len(all_cust)} clientes atendidos, "
              f"capacidades respeitadas.")
    else:
        for e in errors:
            print(e)

    return valid


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CVRP Brute Force Solver — resolve instâncias Solomon-like por força bruta."
    )
    parser.add_argument("file", nargs="?", help="Arquivo .txt da instância")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostra detalhes durante a busca")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Resolve todos os arquivos .txt no diretório atual")
    args = parser.parse_args()

    files_to_solve = []

    if args.all:
        files_to_solve = sorted(
            f for f in os.listdir(".") if f.endswith(".txt")
        )
        if not files_to_solve:
            print("Nenhum arquivo .txt encontrado no diretório atual.")
            sys.exit(1)
    elif args.file:
        files_to_solve = [args.file]
    else:
        parser.print_help()
        sys.exit(1)

    for fpath in files_to_solve:
        try:
            instance = parse_instance(fpath)
        except Exception as e:
            print(f"\n[ERRO ao ler '{fpath}': {e}]")
            continue

        solution = solve_brute_force(instance, verbose=args.verbose)
        print_solution(solution, instance)
        validate_solution(solution, instance)

    print()


if __name__ == "__main__":
    main()
