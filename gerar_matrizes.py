
import pandas as pd
import matplotlib.pyplot as plt

# Carrega os resultados gerados pelos experimentos
df = pd.read_csv("resultados_mlp_por_fold.csv")

# Seleciona apenas o melhor modelo
melhor = df[df["experimento"] == "2_Neuronios_mais"]

fig, axes = plt.subplots(1, 5, figsize=(19, 4))

for ax, (_, row) in zip(axes, melhor.iterrows()):
    # Linhas: classe real | Colunas: classe prevista
    matriz = [
        [int(row["vp"]), int(row["fn"])],
        [int(row["fp"]), int(row["vn"])]
    ]

    ax.imshow(matriz, cmap="Blues", vmin=0, vmax=80)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i][j]),
                    ha="center", va="center", fontsize=16)

    ax.set_xticks([0, 1], ["Maligno", "Benigno"])
    ax.set_yticks([0, 1], ["Maligno", "Benigno"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Fold {int(row['fold'])}")

plt.suptitle("Matrizes de Confusao - MLP (45, 20)")
plt.tight_layout()
plt.savefig("matrizes_confusao_codespaces.png", dpi=200)
plt.show()

print("Imagem gerada com sucesso!")
