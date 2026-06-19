# -*- coding: utf-8 -*-
"""
=============================================================================
 pseudoinversa.py
 Trabalho de Calculo Numerico - Pseudoinversa de Moore-Penrose
 IFCE - Engenharia de Computacao e Telecomunicacoes
 Prof. Dr. Francisco Aquino
 Autores: Vinicius Menezes Monte e Marcelo Flávio de CarvalhoFrota Porto
=============================================================================

Implementa o calculo da pseudoinversa A+ de Moore-Penrose a partir da
Decomposicao em Valores Singulares (SVD), SEM usar numpy.linalg.pinv
diretamente, e reproduz todos os experimentos numericos discutidos no
documento.

Equivalentes em Scilab (linguagem do livro do Prof. Aquino, 2020):
    [U,S,V] = svd(A)   // decomposicao em valores singulares
    pinv(A)            // pseudoinversa pronta (Cap. 5 do livro)
    rank(A)            // posto numerico
    cond(A)            // numero de condicao

Bibliotecas: numpy, matplotlib, tabulate.
Uso:
    python pseudoinversa.py
Gera no diretorio atual: fig_erro_rank.png, fig_ajuste_curva.png,
fig_condicionamento.png
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                # backend nao interativo (salva em arquivo)
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
except ImportError:                  # fallback simples caso tabulate falte
    def tabulate(rows, headers=(), floatfmt="", tablefmt=""):
        out = ["  ".join(str(h) for h in headers)]
        for r in rows:
            out.append("  ".join(str(c) for c in r))
        return "\n".join(out)

np.random.seed(42)                   # reprodutibilidade
np.set_printoptions(precision=5, suppress=True)


# =========================================================================
# 1. IMPLEMENTACAO MANUAL DA PSEUDOINVERSA VIA SVD
# =========================================================================
def pinv_svd(A, rcond=None, k=None):
    """
    Calcula a pseudoinversa A+ de Moore-Penrose usando a SVD.

    A = U @ Sigma @ V^T  =>  A+ = V @ Sigma+ @ U^T

    onde Sigma+ e obtida invertendo cada valor singular NAO nulo
    (acima de um limiar/threshold) e transpondo.

    Parametros
    ----------
    A     : matriz (m x n)
    rcond : limiar relativo. Valores singulares com sigma_i < rcond*sigma_max
            sao tratados como zero (criterio de "parada"/tolerancia, analogo
            ao criterio de tolerancia dos metodos iterativos). Se None, usa
            o padrao  max(m,n) * eps_maquina.
    k     : se informado, usa SVD TRUNCADA com apenas os k maiores valores
            singulares (ignora rcond). Util para o estudo de erro vs. rank.

    Retorna
    -------
    A_plus : pseudoinversa (n x m)
    info   : dicionario com valores singulares, rank numerico e threshold.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # full_matrices=False -> U (m x r), s (r,), Vt (r x n), r = min(m,n)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    if rcond is None:
        rcond = max(m, n) * np.finfo(float).eps   # threshold padrao
    threshold = rcond * (s[0] if s.size else 0.0)

    # vetor de inversos 1/sigma_i, zerando os valores singulares "pequenos"
    s_inv = np.zeros_like(s)
    if k is None:
        mask = s > threshold
        s_inv[mask] = 1.0 / s[mask]
        rank_num = int(mask.sum())
    else:
        kk = min(k, s.size)
        s_inv[:kk] = np.where(s[:kk] > 0, 1.0 / s[:kk], 0.0)
        rank_num = kk

    # A+ = V * Sigma+ * U^T  (Sigma+ = diag(s_inv))
    A_plus = (Vt.T * s_inv) @ U.T

    info = {"valores_singulares": s, "rank_numerico": rank_num,
            "threshold": threshold}
    return A_plus, info


# =========================================================================
# 2. VALIDACAO: comparacao com numpy.linalg.pinv e as 4 condicoes de M-P
# =========================================================================
def condicoes_moore_penrose(A, Ap, tol=1e-9):
    """Verifica as 4 condicoes que definem unicamente A+.
       (1) A A+ A = A
       (2) A+ A A+ = A+
       (3) (A A+)^T = A A+      (A A+ simetrica)
       (4) (A+ A)^T = A+ A      (A+ A simetrica)
    """
    c1 = np.allclose(A @ Ap @ A, A, atol=tol)
    c2 = np.allclose(Ap @ A @ Ap, Ap, atol=tol)
    c3 = np.allclose((A @ Ap).T, A @ Ap, atol=tol)
    c4 = np.allclose((Ap @ A).T, Ap @ A, atol=tol)
    return c1, c2, c3, c4


def validar():
    print("=" * 74)
    print(" 2. VALIDACAO DA IMPLEMENTACAO")
    print("=" * 74)
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],     # linha = 2*linha2 - linha1  -> rank 2
                  [2.0, 1.0, 0.0]])
    Ap_meu, info = pinv_svd(A)
    Ap_np = np.linalg.pinv(A)

    erro = np.linalg.norm(Ap_meu - Ap_np)
    print(f"Matriz A de teste {A.shape}, posto numerico = {info['rank_numerico']}")
    print(f"Valores singulares: {info['valores_singulares']}")
    print(f"||A+_manual - A+_numpy||_F = {erro:.3e}  "
          f"(coincidem? {np.allclose(Ap_meu, Ap_np)})")

    c = condicoes_moore_penrose(A, Ap_meu)
    nomes = ["AA+A = A", "A+AA+ = A+", "(AA+)^T = AA+", "(A+A)^T = A+A"]
    print("\nCondicoes de Moore-Penrose (implementacao manual):")
    for nome, ok in zip(nomes, c):
        print(f"   {nome:<16}: {'OK' if ok else 'FALHOU'}")
    print(f"Todas as 4 condicoes satisfeitas: {all(c)}")
    return all(c) and np.allclose(Ap_meu, Ap_np)


# =========================================================================
# 3. EXEMPLO 1 - SISTEMA SOBREDETERMINADO (4x2): MINIMOS QUADRADOS
# =========================================================================
def exemplo_sobredeterminado():
    print("\n" + "=" * 74)
    print(" 3. EXEMPLO 1 - SISTEMA SOBREDETERMINADO (4x2)  ->  MINIMOS QUADRADOS")
    print("=" * 74)
    # 4 equacoes, 2 incognitas: em geral incompativel (sem solucao exata).
    A = np.array([[1.0, 0.0],
                  [1.0, 1.0],
                  [1.0, 2.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0, 2.0, 4.0])    # pontos quase sobre uma reta

    Ap, _ = pinv_svd(A)
    x = Ap @ b                            # solucao de minimos quadrados
    r = A @ x - b
    print(f"A (4x2):\n{A}")
    print(f"b = {b}")
    print(f"x* = A+ b = {x}   (intercepto={x[0]:.4f}, inclinacao={x[1]:.4f})")
    print(f"residuo  A x* - b = {r}")
    print(f"||A x* - b||_2 = {np.linalg.norm(r):.6f}  (menor residuo possivel)")
    # confirma que e ortogonal ao espaco coluna (equacao normal)
    print(f"A^T (A x* - b) = {A.T @ r}  (~0 -> condicao de minimos quadrados)")
    return A, b, x


# =========================================================================
# 4. EXEMPLO 2 - SISTEMA SUBDETERMINADO (2x4): MINIMA NORMA
# =========================================================================
def exemplo_subdeterminado():
    print("\n" + "=" * 74)
    print(" 4. EXEMPLO 2 - SISTEMA SUBDETERMINADO (2x4)  ->  SOLUCAO DE MINIMA NORMA")
    print("=" * 74)
    # 2 equacoes, 4 incognitas: infinitas solucoes. A+ escolhe a de menor norma.
    A = np.array([[1.0, 1.0, 1.0, 1.0],
                  [1.0, 2.0, 3.0, 4.0]])
    b = np.array([6.0, 16.0])

    Ap, _ = pinv_svd(A)
    x = Ap @ b
    print(f"A (2x4):\n{A}")
    print(f"b = {b}")
    print(f"x* = A+ b = {x}")
    print(f"||A x* - b||_2 = {np.linalg.norm(A @ x - b):.3e}  (solucao exata)")
    print(f"||x*||_2 = {np.linalg.norm(x):.6f}  (MINIMA entre todas as solucoes)")

    # comprovacao: qualquer outra solucao tem norma maior.
    # vetor do espaco nulo de A (Av = 0); x* + t v tambem resolve, com norma maior.
    from numpy.linalg import svd
    _, _, Vt = svd(A)
    v = Vt[-1]                       # ultimo vetor singular -> base do nucleo
    for t in (-1.0, 0.5, 1.0):
        xx = x + t * v
        print(f"   x* + ({t:+.1f})v : ||Ax-b||={np.linalg.norm(A@xx-b):.1e}, "
              f"||x||={np.linalg.norm(xx):.4f}")
    return A, b, x


# =========================================================================
# 5. ANALISE DE RANK E ERRO - SVD TRUNCADA: ||A x_k - b|| vs k
# =========================================================================
def analise_rank_erro():
    print("\n" + "=" * 74)
    print(" 5. ANALISE DE RANK x ERRO (SVD TRUNCADA)")
    print("=" * 74)
    # sistema sobredeterminado com valores singulares decrescentes.
    m, n = 12, 6
    rng = np.random.default_rng(7)
    # cria A com espectro controlado (valores singulares decaem ~geometricamente)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sig = np.array([10.0, 6.0, 3.0, 1.2, 0.4, 0.08])
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(sig)
    A = U @ S @ V.T
    x_true = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.7])
    b = A @ x_true + 0.01 * rng.standard_normal(m)   # com pequeno ruido

    linhas, erros = [], []
    for k in range(1, n + 1):
        Ap_k, _ = pinv_svd(A, k=k)
        x_k = Ap_k @ b
        res = np.linalg.norm(A @ x_k - b)
        erro_sol = np.linalg.norm(x_k - x_true)
        linhas.append([k, f"{sig[k-1]:.3f}", f"{res:.5f}", f"{erro_sol:.5f}",
                       f"{np.linalg.norm(x_k):.4f}"])
        erros.append(res)

    print(tabulate(linhas,
                   headers=["k", "sigma_k", "||A x_k - b||",
                            "||x_k - x_real||", "||x_k||"],
                   tablefmt="github"))

    # grafico erro vs k
    plt.figure(figsize=(7, 4.4))
    ks = list(range(1, n + 1))
    plt.semilogy(ks, erros, "o-", color="#1f3b73", linewidth=2, markersize=7)
    plt.xlabel("k  (valores singulares retidos na SVD truncada)")
    plt.ylabel(r"$\|A x_k - b\|_2$  (escala log)")
    plt.title("Erro de reconstrução vs. posto k da SVD truncada")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fig_erro_rank.png", dpi=150)
    plt.close()
    print(">> figura salva: fig_erro_rank.png")
    return ks, erros


# =========================================================================
# 6. APLICACAO EM COMPUTACAO GRAFICA - AJUSTE POLINOMIAL (MIN. QUADRADOS)
# =========================================================================
def ajuste_polinomial():
    print("\n" + "=" * 74)
    print(" 6. APLICACAO (COMPUTACAO GRAFICA) - AJUSTE DE CURVA POLINOMIAL")
    print("=" * 74)
    # conjunto de pontos 2D "ruidosos" amostrados de uma curva suave
    rng = np.random.default_rng(3)
    x = np.linspace(0, 2 * np.pi, 18)
    y = np.sin(x) + 0.5 * np.cos(0.5 * x)
    y_ruido = y + 0.15 * rng.standard_normal(x.size)

    grau = 5
    # matriz de Vandermonde -> sistema sobredeterminado V c = y
    Vd = np.vander(x, grau + 1, increasing=True)
    Vp, info = pinv_svd(Vd)
    coef = Vp @ y_ruido            # coeficientes via minimos quadrados (A+ b)

    # avaliacao do polinomio ajustado
    xx = np.linspace(x.min(), x.max(), 300)
    Vxx = np.vander(xx, grau + 1, increasing=True)
    yy = Vxx @ coef
    rmse = np.sqrt(np.mean((Vd @ coef - y_ruido) ** 2))
    print(f"Pontos: {x.size} | grau do polinomio: {grau} "
          f"| posto da Vandermonde: {info['rank_numerico']}")
    print(f"Coeficientes (c0..c{grau}): {coef}")
    print(f"RMSE do ajuste: {rmse:.4f}")

    plt.figure(figsize=(7, 4.4))
    plt.scatter(x, y_ruido, color="#c0392b", zorder=5, label="pontos medidos")
    plt.plot(xx, yy, color="#1f3b73", linewidth=2,
             label=f"polinômio grau {grau} (mín. quadrados via $A^+$)")
    plt.plot(xx, np.sin(xx) + 0.5 * np.cos(0.5 * xx), "--",
             color="#555", alpha=0.7, label="curva geradora")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Ajuste de curva por mínimos quadrados com a pseudoinversa")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fig_ajuste_curva.png", dpi=150)
    plt.close()
    print(">> figura salva: fig_ajuste_curva.png")
    return coef


# =========================================================================
# 7. ANALISE DO NUMERO DE CONDICAO  kappa(A) vs ERRO
# =========================================================================
def analise_condicionamento():
    print("\n" + "=" * 74)
    print(" 7. NUMERO DE CONDICAO  kappa(A) = sigma_max / sigma_min  vs  ERRO")
    print("=" * 74)
    n = 6
    rng = np.random.default_rng(11)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    x_true = rng.standard_normal(n)

    kappas = np.logspace(0, 10, 11)      # 1, 10, ... 1e10
    linhas, erros_rel = [], []
    delta = 1e-8                         # perturbacao relativa em b
    for kap in kappas:
        # valores singulares de sigma_max=1 ate sigma_min=1/kap
        sig = np.logspace(0, -np.log10(kap), n)
        A = U @ np.diag(sig) @ V.T
        b = A @ x_true
        # perturba b e mede como o erro em x e amplificado
        db = delta * np.linalg.norm(b) * rng.standard_normal(n)
        db = db / np.linalg.norm(db) * delta * np.linalg.norm(b)
        Ap, _ = pinv_svd(A, rcond=1e-15)
        x_pert = Ap @ (b + db)
        err_rel = np.linalg.norm(x_pert - x_true) / np.linalg.norm(x_true)
        erros_rel.append(err_rel)
        linhas.append([f"{kap:.0e}", f"{np.linalg.norm(db)/np.linalg.norm(b):.1e}",
                       f"{err_rel:.2e}"])

    print(tabulate(linhas,
                   headers=["kappa(A)", "erro rel. em b", "erro rel. em x"],
                   tablefmt="github"))

    plt.figure(figsize=(7, 4.4))
    plt.loglog(kappas, erros_rel, "s-", color="#1f3b73", linewidth=2, markersize=7,
               label="erro relativo na solucao")
    plt.loglog(kappas, delta * kappas, "--", color="#c0392b",
               label=r"limite teorico $\kappa(A)\cdot\delta$")
    plt.xlabel(r"número de condição  $\kappa(A)$")
    plt.ylabel("erro relativo em x")
    plt.title("Impacto do condicionamento na estabilidade numérica")
    plt.legend(fontsize=9); plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fig_condicionamento.png", dpi=150)
    plt.close()
    print(">> figura salva: fig_condicionamento.png")
    return kappas, erros_rel


# =========================================================================
# PROGRAMA PRINCIPAL
# =========================================================================
if __name__ == "__main__":
    print("\n" + "#" * 74)
    print("#  PSEUDOINVERSA DE MOORE-PENROSE VIA SVD - EXPERIMENTOS NUMERICOS")
    print("#" * 74)
    ok = validar()
    exemplo_sobredeterminado()
    exemplo_subdeterminado()
    analise_rank_erro()
    ajuste_polinomial()
    analise_condicionamento()
    print("\n" + "#" * 74)
    print(f"#  CONCLUIDO. Validacao geral: {'SUCESSO' if ok else 'FALHA'}")
    print("#  Figuras: fig_erro_rank.png, fig_ajuste_curva.png, fig_condicionamento.png")
    print("#" * 74)
