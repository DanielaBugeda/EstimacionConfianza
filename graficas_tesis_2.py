import math
import pandas as pd
import matplotlib.pyplot as plt
import funciones as fn
import numpy as np
import seaborn

# Exportar tabla de estimaciones
estimaciones = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/estimaciones.txt', dtype=float)

# Divisiones por número de aristas
estimaciones1 = estimaciones[(estimaciones['distFS'] == 2)].reset_index()
estimaciones2 = estimaciones[(estimaciones['distFS'] == 3)].reset_index()
estimaciones3 = estimaciones[(estimaciones['distFS'] == 4)].reset_index()
estimaciones4 = estimaciones[(estimaciones['distFS'] == 5)].reset_index()
estimaciones5 = estimaciones[(estimaciones['distFS'] == 6)].reset_index()


# # Error absoluto PROM
estimaciones1['error_absoluto_prom'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_prom'])
estimaciones2['error_absoluto_prom'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_prom'])
estimaciones3['error_absoluto_prom'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_prom'])
estimaciones4['error_absoluto_prom'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_prom'])
estimaciones5['error_absoluto_prom'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_prom'])

# Error absoluto TT
estimaciones1['error_absoluto_tt'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_tt'])
estimaciones2['error_absoluto_tt'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_tt'])
estimaciones3['error_absoluto_tt'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_tt'])
estimaciones4['error_absoluto_tt'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_tt'])
estimaciones5['error_absoluto_tt'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_tt'])

# Error absoluto EDK1
estimaciones1['error_absoluto_edkp1'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_edkp1'])
estimaciones2['error_absoluto_edkp1'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_edkp1'])
estimaciones3['error_absoluto_edkp1'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_edkp1'])
estimaciones4['error_absoluto_edkp1'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_edkp1'])
estimaciones5['error_absoluto_edkp1'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_edkp1'])

# Error absoluto EDKP2
estimaciones1['error_absoluto_edkp2'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_edkp2'])
estimaciones2['error_absoluto_edkp2'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_edkp2'])
estimaciones3['error_absoluto_edkp2'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_edkp2'])
estimaciones4['error_absoluto_edkp2'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_edkp2'])
estimaciones5['error_absoluto_edkp2'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_edkp2'])

# Resultados
eje_x = ['2','3','4','5','6']
result = pd.DataFrame()

result['eje_x'] = eje_x
result['error_absoluto_prom'] = 0
result['error_absoluto_tt'] = 0
result['error_absoluto_edkp1'] = 0
result['error_absoluto_edkp2'] = 0

result.loc[0, 'error_absoluto_prom'] = sum(estimaciones1['error_absoluto_prom']) / len(estimaciones1)
result.loc[1, 'error_absoluto_prom'] = sum(estimaciones2['error_absoluto_prom']) / len(estimaciones2)
result.loc[2, 'error_absoluto_prom'] = sum(estimaciones3['error_absoluto_prom']) / len(estimaciones3)
result.loc[3, 'error_absoluto_prom'] = sum(estimaciones4['error_absoluto_prom']) / len(estimaciones4)
result.loc[4, 'error_absoluto_prom'] = sum(estimaciones5['error_absoluto_prom']) / len(estimaciones5)

result.loc[0, 'error_absoluto_tt'] = sum(estimaciones1['error_absoluto_tt']) / len(estimaciones1)
result.loc[1, 'error_absoluto_tt'] = sum(estimaciones2['error_absoluto_tt']) / len(estimaciones2)
result.loc[2, 'error_absoluto_tt'] = sum(estimaciones3['error_absoluto_tt']) / len(estimaciones3)
result.loc[3, 'error_absoluto_tt'] = sum(estimaciones4['error_absoluto_tt']) / len(estimaciones4)
result.loc[4, 'error_absoluto_tt'] = sum(estimaciones5['error_absoluto_tt']) / len(estimaciones5)

result.loc[0, 'error_absoluto_edkp1'] = sum(estimaciones1['error_absoluto_edkp1']) / len(estimaciones1)
result.loc[1, 'error_absoluto_edkp1'] = sum(estimaciones2['error_absoluto_edkp1']) / len(estimaciones2)
result.loc[2, 'error_absoluto_edkp1'] = sum(estimaciones3['error_absoluto_edkp1']) / len(estimaciones3)
result.loc[3, 'error_absoluto_edkp1'] = sum(estimaciones4['error_absoluto_edkp1']) / len(estimaciones4)
result.loc[4, 'error_absoluto_edkp1'] = sum(estimaciones5['error_absoluto_edkp1']) / len(estimaciones5)

result.loc[0, 'error_absoluto_edkp2'] = sum(estimaciones1['error_absoluto_edkp2']) / len(estimaciones1)
result.loc[1, 'error_absoluto_edkp2'] = sum(estimaciones2['error_absoluto_edkp2']) / len(estimaciones2)
result.loc[2, 'error_absoluto_edkp2'] = sum(estimaciones3['error_absoluto_edkp2']) / len(estimaciones3)
result.loc[3, 'error_absoluto_edkp2'] = sum(estimaciones4['error_absoluto_edkp2']) / len(estimaciones4)
result.loc[4, 'error_absoluto_edkp2'] = sum(estimaciones5['error_absoluto_edkp2']) / len(estimaciones5)

# print(result)

x = result['eje_x']
y1 = result['error_absoluto_prom']
y2 = result['error_absoluto_tt']
y3 = result['error_absoluto_edkp1']
y4 = result['error_absoluto_edkp2']

plt.plot(x, y1, marker='o', label='Primeros Vecinos')
plt.plot(x, y2, marker='o', label='TidalTrust')
plt.plot(x, y3, marker='o', label='GFTrust, ganancia constante')
plt.plot(x, y4, marker='o', label='GFTrust, ganancia decreciente')

plt.xlabel('Distancia del nodo fuente al nodo sumidero')
plt.ylabel('Error Absoluto Medio')
plt.title('Error Absoluto Medio por la distancia en la WoT entre el nodo fuente y el nodo sumidero')

plt.legend()
plt.show()


# Resultados
eje_x = ['2','3','4','5','6']
result = pd.DataFrame()

result['eje_x'] = eje_x
result['rmse_prom'] = 0
result['rmse_tt'] = 0
result['rmse_edkp1'] = 0
result['rmse_tt'] = 0

result.loc[0, 'rmse_prom'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_prom'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_prom'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_prom'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_prom'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_prom'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_prom'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_prom'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_prom'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_prom'], estimaciones5['confianza_real'])

result.loc[0, 'rmse_tt'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_tt'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_tt'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_tt'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_tt'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_tt'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_tt'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_tt'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_tt'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_tt'], estimaciones5['confianza_real'])

result.loc[0, 'rmse_edkp1'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_edkp1'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_edkp1'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_edkp1'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_edkp1'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_edkp1'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_edkp1'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_edkp1'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_edkp1'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_edkp1'], estimaciones5['confianza_real'])

result.loc[0, 'rmse_edkp2'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_edkp2'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_edkp2'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_edkp2'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_edkp2'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_edkp2'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_edkp2'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_edkp2'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_edkp2'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_edkp2'], estimaciones5['confianza_real'])

print(result)

x = result['eje_x']
y1 = result['rmse_prom']
y2 = result['rmse_tt']
y3 = result['rmse_edkp1']
y4 = result['rmse_edkp2']

plt.plot(x, y1, marker='o', label='Primeros Vecinos')
plt.plot(x, y2, marker='o', label='TidalTrust')
plt.plot(x, y3, marker='o', label='GFTrust, ganancia constante')
plt.plot(x, y4, marker='o', label='GFTrust, ganancia decreciente')

plt.xlabel('Distancia del nodo fuente al nodo sumidero')
plt.ylabel('Raíz del Error Cuadrático Medio')
plt.title('Raíz del Error Cuadrático Medio por la distancia en la WoT entre el nodo fuente y el nodo sumidero')

plt.legend()
plt.show()


# Error de estimación
# Error PROM
estimaciones1['error_estimacion_prom'] = estimaciones1['confianza_real'] - estimaciones1['confianza_prom']
estimaciones2['error_estimacion_prom'] = estimaciones2['confianza_real'] - estimaciones2['confianza_prom']
estimaciones3['error_estimacion_prom'] = estimaciones3['confianza_real'] - estimaciones3['confianza_prom']
estimaciones4['error_estimacion_prom'] = estimaciones4['confianza_real'] - estimaciones4['confianza_prom']
estimaciones5['error_estimacion_prom'] = estimaciones5['confianza_real'] - estimaciones5['confianza_prom']

# Error TT
estimaciones1['error_estimacion_tt'] = estimaciones1['confianza_real'] - estimaciones1['confianza_tt']
estimaciones2['error_estimacion_tt'] = estimaciones2['confianza_real'] - estimaciones2['confianza_tt']
estimaciones3['error_estimacion_tt'] = estimaciones3['confianza_real'] - estimaciones3['confianza_tt']
estimaciones4['error_estimacion_tt'] = estimaciones4['confianza_real'] - estimaciones4['confianza_tt']
estimaciones5['error_estimacion_tt'] = estimaciones5['confianza_real'] - estimaciones5['confianza_tt']

# Error EDK1
estimaciones1['error_estimacion_edkp1'] = estimaciones1['confianza_real'] - estimaciones1['confianza_edkp1']
estimaciones2['error_estimacion_edkp1'] = estimaciones2['confianza_real'] - estimaciones2['confianza_edkp1']
estimaciones3['error_estimacion_edkp1'] = estimaciones3['confianza_real'] - estimaciones3['confianza_edkp1']
estimaciones4['error_estimacion_edkp1'] = estimaciones4['confianza_real'] - estimaciones4['confianza_edkp1']
estimaciones5['error_estimacion_edkp1'] = estimaciones5['confianza_real'] - estimaciones5['confianza_edkp1']

# Error EDKP2
estimaciones1['error_estimacion_edkp2'] = estimaciones1['confianza_real'] - estimaciones1['confianza_edkp2']
estimaciones2['error_estimacion_edkp2'] = estimaciones2['confianza_real'] - estimaciones2['confianza_edkp2']
estimaciones3['error_estimacion_edkp2'] = estimaciones3['confianza_real'] - estimaciones3['confianza_edkp2']
estimaciones4['error_estimacion_edkp2'] = estimaciones4['confianza_real'] - estimaciones4['confianza_edkp2']
estimaciones5['error_estimacion_edkp2'] = estimaciones5['confianza_real'] - estimaciones5['confianza_edkp2']

medianas =[]
medias = []

# Crear el diagrama de cajas
data = [estimaciones1['error_estimacion_prom'], estimaciones1['error_estimacion_tt'], estimaciones1['error_estimacion_edkp1'], estimaciones1['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('Distancia de 2 entre f-s')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4], medians, marker='o', color='r', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
means = [np.mean(d) for d in data]
plt.plot([1, 2, 3, 4], means, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylim(-1.2, 1.2)
plt.show()

medianas.append(medians)
medias.append(means)

data = [estimaciones2['error_estimacion_prom'], estimaciones2['error_estimacion_tt'], estimaciones2['error_estimacion_edkp1'], estimaciones2['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('Distancia de 3 entre f-s')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4], medians, marker='o', color='r', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
means = [np.mean(d) for d in data]
plt.plot([1, 2, 3, 4], means, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylim(-1.2, 1.2)
plt.show()

medianas.append(medians)
medias.append(means)

data = [estimaciones3['error_estimacion_prom'], estimaciones3['error_estimacion_tt'], estimaciones3['error_estimacion_edkp1'], estimaciones3['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('Distancia de 4 entre f-s')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4], medians, marker='o', color='r', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
means = [np.mean(d) for d in data]
plt.plot([1, 2, 3, 4], means, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylim(-1.2, 1.2)
plt.show()

medianas.append(medians)
medias.append(means)

data = [estimaciones4['error_estimacion_prom'], estimaciones4['error_estimacion_tt'], estimaciones4['error_estimacion_edkp1'], estimaciones4['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('Distancia de 5 entre f-s')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4], medians, marker='o', color='r', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
means = [np.mean(d) for d in data]
plt.plot([1, 2, 3, 4], means, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylim(-1.2, 1.2)
plt.show()

medianas.append(medians)
medias.append(means)

data = [estimaciones5['error_estimacion_prom'], estimaciones5['error_estimacion_tt'], estimaciones5['error_estimacion_edkp1'], estimaciones5['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('Distancia de 6 entre f-s')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4], medians, marker='o', color='r', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
means = [np.mean(d) for d in data]
plt.plot([1, 2, 3, 4], means, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.ylim(-1.2, 1.2)
plt.show()

medianas.append(medians)
medias.append(means)

# Distribución de los valores reales - histograma
estimaciones['confianza_real'].hist(bins=10, color = "green")
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
plt.title('Distribución de los valores de confianza en el subconjunto utilizado de la red original')
plt.show()

plt.boxplot(estimaciones['confianza_real'], vert=True, patch_artist=True, boxprops = dict(facecolor = "green"))
plt.xticks([1], ['Valores de confianza en el subconjunto utilizado de la red original'])
plt.ylabel('Valores de confianza')
plt.show()

medianas_res = pd.DataFrame(medianas, columns=['Primeros Vecinos', 'TidalTrust', 'GFTrust constante', 'GFTrust decreciente'])
medias_res = pd.DataFrame(medias, columns=['Primeros Vecinos', 'TidalTrust', 'GFTrust constante', 'GFTrust decreciente'])

print(medianas_res)
print(medias_res)

data = [estimaciones['confianza_real'], estimaciones['confianza_prom'], estimaciones['confianza_tt'], estimaciones['confianza_edkp1'], estimaciones['confianza_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightgreen"), medianprops = dict(color = "green"))
plt.xticks([1, 2, 3, 4, 5], ['Confianza \nReal', 'Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Valores de confianza')
medians = [np.median(d) for d in data]
plt.plot([1, 2, 3, 4, 5], medians, marker='o', color='g', linestyle='-', linewidth=1, zorder=3)
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='th=0.5')
plt.axhline(y=0.6, color='red', linestyle='--', linewidth=1, label='th=0.6')
plt.axhline(y=0.645, color='magenta', linestyle='--', linewidth=1, label='th=0.6245')
plt.axhline(y=0.7, color='blue', linestyle='--', linewidth=1, label='th=0.7')
plt.axhline(y=0.8, color='black', linestyle='--', linewidth=1, label='th=0.8')
plt.ylim(-0.2, 1.2)
plt.legend()
plt.show()

plt.hist(data, bins=30, color=['blue', 'green', 'red', 'yellow', 'purple'], label=['Confianza \nReal', 'Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
# plt.title('Histogram of Multiple Groups')
plt.legend()
plt.show()
print(len(estimaciones))

dataR = pd.DataFrame({"Modelo": ["Confianza Real"]*len(estimaciones), "Estimacion": estimaciones['confianza_real']})
dataP = pd.DataFrame({"Modelo": ["Primeros \nVecinos"]*len(estimaciones), "Estimacion": estimaciones['confianza_prom']})
dataT = pd.DataFrame({"Modelo": ["TidalTrust"]*len(estimaciones), "Estimacion": estimaciones['confianza_tt']})
dataE1 = pd.DataFrame({"Modelo": ["GFTrust, \nconstante"]*len(estimaciones), "Estimacion": estimaciones['confianza_edkp1']})
dataE2 = pd.DataFrame({"Modelo": ["GFTrust, \ndecreciente"]*len(estimaciones), "Estimacion": estimaciones['confianza_edkp2']})

data2 = pd.concat([dataR, dataP, dataT, dataE1, dataE2], ignore_index=True)
print(data2)

custom_palette = ['gray', 'lightblue', 'orange', 'lightgreen', 'red']
seaborn.violinplot(x='Modelo', y='Estimacion', data=data2, palette=custom_palette)
plt.ylim(0.2,1.2)
plt.show()

# Distribución de los valores reales - histograma
plt.subplot(2, 2, 1)
plt.hist([estimaciones['confianza_real'], estimaciones['confianza_prom']], bins=10, color=['gray', '#1f77b4'], label=['Confianza Real', 'Primeros Vecinos'])
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
plt.ylim(0,400)
plt.legend(loc='upper left')

plt.subplot(2, 2, 2)
plt.hist([estimaciones['confianza_real'], estimaciones['confianza_tt']], bins=10, color=['gray', '#ff7f0e'], label=['Confianza Real', 'TidalTrust'])
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
plt.ylim(0,400)
plt.legend(loc='upper left')

plt.subplot(2, 2, 3)
plt.hist([estimaciones['confianza_real'], estimaciones['confianza_edkp1']], bins=10, color=['gray', '#2ca02c'], label=['Confianza Real', 'GFTrust, constante'])
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
plt.ylim(0,400)
plt.legend(loc='upper left')

plt.subplot(2, 2, 4)
plt.hist([estimaciones['confianza_real'], estimaciones['confianza_edkp2']], bins=10, color=['gray', '#d62728'], label=['Confianza Real', 'GFTrust, decreciente'])
plt.xlabel('Valores de confianza')
plt.ylabel('Frecuencia')
plt.ylim(0,400)
plt.legend(loc='upper left')

plt.show()