import math
import pandas as pd
import matplotlib.pyplot as plt
import funciones as fn
import numpy as np

# Exportar tabla de estimaciones
estimaciones = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/estimaciones.txt', dtype=float)

estimaciones['error_estimacion_prom'] = estimaciones['confianza_real'] - estimaciones['confianza_prom']
estimaciones['error_estimacion_TT'] = estimaciones['confianza_real'] - estimaciones['confianza_tt']
estimaciones['error_estimacion_EDKP1'] = estimaciones['confianza_real'] - estimaciones['confianza_edkp1']
estimaciones['error_estimacion_EDKP2'] = estimaciones['confianza_real'] - estimaciones['confianza_edkp2']

data = [estimaciones['error_estimacion_prom'], estimaciones['error_estimacion_TT'], estimaciones['error_estimacion_EDKP1'], estimaciones['error_estimacion_EDKP2']]
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4], medians, marker='o', color='r', linestyle='-', linewidth=1, zorder=3)
means = [np.mean(d) for d in data]
plt.plot([1, 2, 3, 4], means, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylim(-1.2,1.2)
plt.show()

print(" -------- RMSE ------")
print(fn.RMSE(len(estimaciones), estimaciones['confianza_prom'], estimaciones['confianza_real']))
print(fn.RMSE(len(estimaciones), estimaciones['confianza_tt'], estimaciones['confianza_real']))
print(fn.RMSE(len(estimaciones), estimaciones['confianza_edkp1'], estimaciones['confianza_real']))
print(fn.RMSE(len(estimaciones), estimaciones['confianza_edkp2'], estimaciones['confianza_real']))

estimaciones['abs_prom'] = abs(estimaciones['confianza_prom'] - estimaciones['confianza_real'])
estimaciones['abs_tt'] = abs(estimaciones['confianza_tt'] - estimaciones['confianza_real'])
estimaciones['abs_edkp1']= abs(estimaciones['confianza_edkp1'] - estimaciones['confianza_real'])
estimaciones['abs_edkp2'] = abs(estimaciones['confianza_edkp2'] - estimaciones['confianza_real'])

print(" -------- MAE ------")
print(estimaciones['abs_prom'].sum() / len(estimaciones))
print(estimaciones['abs_tt'].sum() / len(estimaciones))
print(estimaciones['abs_edkp1'].sum() / len(estimaciones))
print(estimaciones['abs_edkp2'].sum() / len(estimaciones))

print(estimaciones['error_estimacion_TT'].mean())

# Exportar tabla de estimaciones
bbdd = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/BBDD_Bitcoin.txt', dtype=float)

# Distribución de los valores reales - histograma
bbdd['trust'].hist(bins=10)
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
plt.title('Distribución de los valores de confianza en la red original')
plt.show()

plt.boxplot(bbdd['trust'], vert=True, patch_artist=True)
plt.xticks([1], ['Valores de confianza en la red original'])
plt.ylabel('Valores de confianza')
plt.show()
