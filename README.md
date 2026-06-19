# Pseudoinversa de Moore-Penrose via SVD

Trabalho de **Cálculo Numérico** — Instituto Federal do Ceará (IFCE), cursos de
Engenharia de Computação e de Telecomunicações.

Implementação da pseudoinversa $A^{+}$ de Moore-Penrose a partir da Decomposição
em Valores Singulares (SVD), **sem usar `numpy.linalg.pinv` diretamente**, acompanhada
de uma monografia que discute a história do cálculo numérico e a fundamentação,
o algoritmo e as aplicações da pseudoinversa.

- **Autores:** Vinícius Menezes Monte e Marcelo Flávio de Carvalho Frota Porto
- **Orientador:** Prof. Dr. Francisco Aquino
- **Fortaleza — Ceará, 2026**

---

## Sobre o projeto

A pseudoinversa estende a ideia de inversa para matrizes retangulares ou singulares,
onde a inversa clássica $A^{-1}$ não existe. De uma só vez, ela fornece:

- a solução de **mínimos quadrados** de sistemas sobredeterminados ($m > n$);
- a solução de **mínima norma** de sistemas subdeterminados ($m < n$).

O cálculo parte da SVD, $A = U\,\Sigma\,V^{\mathsf{T}}$, de onde
$A^{+} = V\,\Sigma^{+}\,U^{\mathsf{T}}$, invertendo apenas os valores singulares acima
de um limiar (*threshold*) que define o posto numérico da matriz — papel análogo ao do
critério de tolerância dos métodos iterativos.

## Estrutura do repositório

```
.
├── docs/
│   ├── main.tex                          # fonte LaTeX da monografia (ABNT)
│   ├── Trabalho_de_Cálculo_Numérico.pdf  # monografia compilada
│   ├── fig_erro_rank.png                 # erro vs. posto k (SVD truncada)
│   ├── fig_ajuste_curva.png              # ajuste polinomial por mínimos quadrados
│   └── fig_condicionamento.png           # erro relativo vs. número de condição
└── src/
    ├── pseudoinversa.py                   # implementação + experimentos numéricos
    └── env/                               # ambiente virtual Python (local)
```

## O que o código faz

O script `src/pseudoinversa.py` implementa `pinv_svd(A, rcond=None, k=None)` e reproduz
todos os experimentos discutidos na monografia:

1. **Validação** — compara $A^{+}$ com `numpy.linalg.pinv` e verifica numericamente as
   quatro condições de Moore-Penrose.
2. **Sistema sobredeterminado (4×2)** — solução de mínimos quadrados.
3. **Sistema subdeterminado (2×4)** — solução de mínima norma.
4. **Análise de posto × erro** — SVD truncada, $\lVert A x_k - b\rVert$ versus $k$.
5. **Aplicação (Computação Gráfica)** — ajuste de curva polinomial via Vandermonde.
6. **Número de condição** — impacto de $\kappa(A)$ na estabilidade numérica.

Ao final, o script imprime os resultados no terminal e salva três figuras no diretório
de execução: `fig_erro_rank.png`, `fig_ajuste_curva.png` e `fig_condicionamento.png`.

## Requisitos

- Python 3.13 (testado com 3.13.5)
- [`numpy`](https://numpy.org/) — núcleo do cálculo
- [`matplotlib`](https://matplotlib.org/) — geração das figuras
- [`tabulate`](https://pypi.org/project/tabulate/) — formatação das tabelas (opcional;
  há um *fallback* embutido caso não esteja instalado)

## Como executar

A partir da raiz do projeto:

```bash
# 1. (opcional) criar e ativar um ambiente virtual
python -m venv env
# Windows (PowerShell)
env\Scripts\Activate.ps1
# Linux / macOS
source env/bin/activate

# 2. instalar as dependências
pip install numpy matplotlib tabulate

# 3. rodar os experimentos
cd src
python pseudoinversa.py
```

As três figuras são geradas no diretório atual. Para reproduzir as imagens usadas na
monografia, copie-as para `docs/` ou execute o script a partir dessa pasta.

## Compilar a monografia

O documento em `docs/main.tex` segue formatação ABNT (Times 12pt, espaçamento 1,5).
Compile com `pdflatex` (três passagens, para resolver o sumário):

```bash
cd docs
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

> As figuras referenciadas pela monografia devem estar em `docs/` antes da compilação.

## Referências principais

- AQUINO, F. J. A. de. *Tópicos de métodos numéricos com Scilab*. Rio de Janeiro: PoD, 2020.
- GOLUB, G. H.; VAN LOAN, C. F. *Matrix Computations*. 4. ed. Johns Hopkins, 2013.
- PENROSE, R. A generalized inverse for matrices. *Math. Proc. Cambridge Phil. Soc.*, v. 51, 1955.
- MOORE, E. H. On the reciprocal of the general algebraic matrix. *Bull. AMS*, v. 26, 1920.

A lista completa de referências está na monografia.
