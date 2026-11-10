
"""Experimentos MLP - UTFPR, baseados em MLP_Breast_Cancer_Classification.py.
Mantém pipeline, KFold e seed da aula. Uma alteração por experimento.
"""
import warnings
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target  # 0 = maligno; 1 = benigno
kf = KFold(n_splits=5, shuffle=True, random_state=42)
experimentos = [
    ('1_Base', (30,15), 0.0001),
    ('2_Neuronios_mais', (45,20), 0.0001),
    ('3_Neuronios_menos', (15,8), 0.0001),
    ('4_Alpha_maior', (30,15), 0.001),
    ('5_Alpha_menor', (30,15), 0.00001),
]
registros=[]
matrizes=[]
with warnings.catch_warnings():
    warnings.simplefilter('ignore', ConvergenceWarning)
    for nome, camadas, alpha in experimentos:
        for fold, (treino, teste) in enumerate(kf.split(X), 1):
            modelo = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=camadas, max_iter=1000, random_state=42, alpha=alpha))
            modelo.fit(X.iloc[treino], y[treino])
            previsto = modelo.predict(X.iloc[teste])
            matriz = confusion_matrix(y[teste], previsto, labels=[0,1])
            # matriz [[VP, FN], [FP, VN]] ao considerar maligno (0) positivo.
            vp, fn, fp, vn = matriz.ravel()
            registros.append(dict(experimento=nome, fold=fold, camadas=str(camadas), alpha=alpha,
                acuracia=accuracy_score(y[teste],previsto),
                recall_maligno=recall_score(y[teste],previsto,pos_label=0,zero_division=0),
                precisao_maligno=precision_score(y[teste],previsto,pos_label=0,zero_division=0),
                f1_maligno=f1_score(y[teste],previsto,pos_label=0,zero_division=0),
                recall_benigno=recall_score(y[teste],previsto,pos_label=1,zero_division=0),
                vp=int(vp),fn=int(fn),fp=int(fp),vn=int(vn),
                iteracoes=modelo.named_steps['mlpclassifier'].n_iter_))
            matrizes.append((nome,fold,matriz))

res=pd.DataFrame(registros)
pasta_saida = Path(__file__).resolve().parent
res.to_csv(pasta_saida / 'resultados_mlp_por_fold.csv', index=False)
cols=['acuracia','recall_maligno','precisao_maligno','f1_maligno','recall_benigno','vp','fn','fp','vn']
resumo=res.groupby('experimento',sort=False)[cols].mean().reset_index()
resumo.to_csv(pasta_saida / 'resumo_experimentos_mlp.csv', index=False)
print(resumo.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
print('\nMATRIZES POR FOLD:')
print(res[['experimento','fold','vp','fn','fp','vn','iteracoes']].to_string(index=False))

print(f'\nArquivos CSV salvos em: {pasta_saida}')
