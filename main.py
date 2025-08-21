# Registro da licença do gurobi

from gurobipy import Env, Model, GRB
import gurobipy as gp
import random
import pandas as pd
import os

import zipfile

# Tem que ter uma licença do gurobi configurada

env = Env()

# Função para garantir cobertura de todos os indices pelos subconjuntos
def garantir_cobertura(S, n, rng):
    presentes = set(e for s in S for e in s)
    faltando = [k for k in range(1, n+1) if k not in presentes]
    for k in faltando:
        idx = rng.randrange(len(S))
        if k not in S[idx]:
            S[idx].append(k)
    return S

# Função para gerar instâncias
def gerar_instancia(n, padrao=1, seed=None):
    if seed is not None:
        random.seed(seed)
    rng = random.Random(seed)
    S = []
    for i in range(n):
        if padrao == 1:
            tam = max(1, int(n**0.5)) #Subconjuntos menores
        elif padrao == 2:
            tam = max(1, n//4) # Subconjuntos médios
        else:
            tam = random.randint(1, n//2) # tamanho aleatório
        elementos = random.sample(range(1, n+1), tam)
        S.append(elementos)

    S = garantir_cobertura(S, n, rng)
    # matriz triangular superior
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            A[i][j] = random.randint(-5, 10)

    return n, S, A

def gerar_instancia_dificil(n, padrao=1, seed=None):
    """
    Gera subconjuntos S e matriz A mais 'conflitantes' conforme o padrão:
      - padrao=1 => subconjuntos menores (≈ sqrt(n)), fortemente sobrepostos
      - padrao=2 => subconjuntos médios (≈ n/4), com bastante sobreposição
      - padrao=3 => tamanhos aleatórios + perturbações

    Além disso cria uma matriz A com pesos mais altos quando (i,j) aparecem juntos em vários S.
    """
    rng = random.Random(seed)
    S = []
    base = rng.sample(range(1, n+1), n//10)

    for i in range(n):
        if padrao == 1:
            tam = max(1, int(n**0.5))
        elif padrao == 2:
            tam = max(1, n//4)
        else:
            tam = rng.randint(1, n//2)

        s_i = rng.sample(base, min(1, len(base)//2))
        perturb = rng.sample(range(1, n+1), rng.randint(1, max(1, tam)))
        s_i = list(set(s_i + perturb))
        S.append(s_i)

    S = garantir_cobertura(S, n, rng)

    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            #count = sum((i+1 in S[k] and j+1 in S[k]) for k in range(n))
            A[i][j] = rng.randint(-100, 100)

    return n, S, A


def salvar_instancia(n, padrao, S, A, pasta="instancias_arrumadas"):
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    caminho = f"{pasta}/instancia_n{n}_padrao{padrao}.txt"
    with open(caminho, "w") as f:
        f.write(f"{n}\n")
        for subset in S:
            f.write(str(len(subset)) + " ")
        f.write("\n")
        for subset in S:    
            f.write(" ".join(map(str, subset)) + "\n")
        for linha in A:
            f.write(" ".join(map(str, linha)) + "\n")
    return caminho

# Resolver versão linearizada (MILP)
def resolver_max_sc_qbf_linear(n, S, A, timelimit=600, env=None, padrao=1):
    m = Model("MAX_SC_QBF_LINEAR", env=env)

    # variáveis xi (binárias)
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # variáveis yij (binárias),  para todos i ≤ j
    y = {}
    for i in range(n):
        for j in range(i, n):
            y[i,j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # Função objetivo: soma A[i][j] * yij
    m.setObjective(
        gp.quicksum(A[i][j] * y[i,j] for i in range(n) for j in range(i, n)),
        GRB.MAXIMIZE
    )

    # Linearização
    for i in range(n):
        for j in range(i, n):
            m.addConstr(y[i,j] <= x[i])
            m.addConstr(y[i,j] <= x[j])
            m.addConstr(y[i,j] >= x[i] + x[j] - 1)

    # Cobertura
    for k in range(1, n+1):
        m.addConstr(gp.quicksum(x[i] for i in range(n) if k in S[i]) >= 1)

    m.setParam("TimeLimit", timelimit)
    #m.write(f"modelo_400_{padrao}.mps")
    m.optimize()

    return {
        "obj": m.ObjVal if m.SolCount > 0 else None,
        "gap": m.MIPGap if m.SolCount > 0 else None,
        "time": m.Runtime
    }

# --- Loop principal com impressão da instância --------------------------------------------
ns = [25, 50, 100, 200, 400]
padroes = [1, 2, 3]

#ns = [25]
#padroes = [1]

resultados = []

for n in ns:
    for padrao in padroes:
        print(f"🔹 Rodando instância n={n}, padrão={padrao}")

        # n_, S, A = gerar_instancia(n, padrao=padrao, seed=42)
        n_, S, A = gerar_instancia_dificil(n, padrao=padrao, seed=42)

        # ✅ Salvar a instância em arquivo
        salvar_instancia(n, padrao, S, A)

        # >>>> imprimir a instância gerada <<<<
        #print(f"n = {n_}")
        #print("S = ", S)
        #print("A = ")
        #for row in A:
        #    print(row)
        #print("-"*40)

        res = resolver_max_sc_qbf_linear(n_, S, A, timelimit=600, env=env, padrao=padrao)
        resultados.append({
            "n": n,
            "padrao": padrao,
            "obj": res["obj"],
            "gap": res["gap"],
            "tempo": res["time"]
        })

df = pd.DataFrame(resultados)
df.to_csv("resultados_milp.csv", index=False)

# Criar ZIP das instâncias
#with zipfile.ZipFile("instancias.zip", 'w') as zipf:
#    for raiz, dirs, arquivos in os.walk("instancias"):
#        for arquivo in arquivos:
#            zipf.write(os.path.join(raiz, arquivo))
