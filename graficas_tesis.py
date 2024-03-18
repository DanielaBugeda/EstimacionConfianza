import math
import pandas as pd
import matplotlib.pyplot as plt
import funciones as fn
import numpy as np

# Exportar tabla de estimaciones
estimaciones = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/estimaciones.txt', dtype=float)

# Divisiones por número de aristas
estimaciones1 = estimaciones[(estimaciones['WoT_aristas'] >= 3) & (estimaciones['WoT_aristas'] <= 16)].reset_index()
estimaciones2 = estimaciones[(estimaciones['WoT_aristas'] >= 17) & (estimaciones['WoT_aristas'] <= 29)].reset_index()
estimaciones3 = estimaciones[(estimaciones['WoT_aristas'] >= 30) & (estimaciones['WoT_aristas'] <= 45)].reset_index()
estimaciones4 = estimaciones[(estimaciones['WoT_aristas'] >= 46) & (estimaciones['WoT_aristas'] <= 65)].reset_index()
estimaciones5 = estimaciones[(estimaciones['WoT_aristas'] >= 66) & (estimaciones['WoT_aristas'] <= 86)].reset_index()
estimaciones6 = estimaciones[(estimaciones['WoT_aristas'] >= 87) & (estimaciones['WoT_aristas'] <= 112)].reset_index()
estimaciones7 = estimaciones[(estimaciones['WoT_aristas'] >= 113) & (estimaciones['WoT_aristas'] <= 154)].reset_index()
estimaciones8 = estimaciones[(estimaciones['WoT_aristas'] >= 155) & (estimaciones['WoT_aristas'] <= 195)].reset_index()
estimaciones9 = estimaciones[(estimaciones['WoT_aristas'] >= 196) & (estimaciones['WoT_aristas'] <= 251)].reset_index()
estimaciones10 = estimaciones[(estimaciones['WoT_aristas'] >= 252) & (estimaciones['WoT_aristas'] <= 422)].reset_index()

# Error absoluto PROM
estimaciones1['error_absoluto_prom'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_prom'])
estimaciones2['error_absoluto_prom'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_prom'])
estimaciones3['error_absoluto_prom'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_prom'])
estimaciones4['error_absoluto_prom'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_prom'])
estimaciones5['error_absoluto_prom'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_prom'])
estimaciones6['error_absoluto_prom'] = abs(estimaciones6['confianza_real'] - estimaciones6['confianza_prom'])
estimaciones7['error_absoluto_prom'] = abs(estimaciones7['confianza_real'] - estimaciones7['confianza_prom'])
estimaciones8['error_absoluto_prom'] = abs(estimaciones8['confianza_real'] - estimaciones8['confianza_prom'])
estimaciones9['error_absoluto_prom'] = abs(estimaciones9['confianza_real'] - estimaciones9['confianza_prom'])
estimaciones10['error_absoluto_prom'] = abs(estimaciones10['confianza_real'] - estimaciones10['confianza_prom'])

# Error absoluto TT
estimaciones1['error_absoluto_tt'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_tt'])
estimaciones2['error_absoluto_tt'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_tt'])
estimaciones3['error_absoluto_tt'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_tt'])
estimaciones4['error_absoluto_tt'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_tt'])
estimaciones5['error_absoluto_tt'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_tt'])
estimaciones6['error_absoluto_tt'] = abs(estimaciones6['confianza_real'] - estimaciones6['confianza_tt'])
estimaciones7['error_absoluto_tt'] = abs(estimaciones7['confianza_real'] - estimaciones7['confianza_tt'])
estimaciones8['error_absoluto_tt'] = abs(estimaciones8['confianza_real'] - estimaciones8['confianza_tt'])
estimaciones9['error_absoluto_tt'] = abs(estimaciones9['confianza_real'] - estimaciones9['confianza_tt'])
estimaciones10['error_absoluto_tt'] = abs(estimaciones10['confianza_real'] - estimaciones10['confianza_tt'])

# Error absoluto EDK1
estimaciones1['error_absoluto_edkp1'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_edkp1'])
estimaciones2['error_absoluto_edkp1'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_edkp1'])
estimaciones3['error_absoluto_edkp1'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_edkp1'])
estimaciones4['error_absoluto_edkp1'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_edkp1'])
estimaciones5['error_absoluto_edkp1'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_edkp1'])
estimaciones6['error_absoluto_edkp1'] = abs(estimaciones6['confianza_real'] - estimaciones6['confianza_edkp1'])
estimaciones7['error_absoluto_edkp1'] = abs(estimaciones7['confianza_real'] - estimaciones7['confianza_edkp1'])
estimaciones8['error_absoluto_edkp1'] = abs(estimaciones8['confianza_real'] - estimaciones8['confianza_edkp1'])
estimaciones9['error_absoluto_edkp1'] = abs(estimaciones9['confianza_real'] - estimaciones9['confianza_edkp1'])
estimaciones10['error_absoluto_edkp1'] = abs(estimaciones10['confianza_real'] - estimaciones10['confianza_edkp1'])

# Error absoluto EDKP2
estimaciones1['error_absoluto_edkp2'] = abs(estimaciones1['confianza_real'] - estimaciones1['confianza_edkp2'])
estimaciones2['error_absoluto_edkp2'] = abs(estimaciones2['confianza_real'] - estimaciones2['confianza_edkp2'])
estimaciones3['error_absoluto_edkp2'] = abs(estimaciones3['confianza_real'] - estimaciones3['confianza_edkp2'])
estimaciones4['error_absoluto_edkp2'] = abs(estimaciones4['confianza_real'] - estimaciones4['confianza_edkp2'])
estimaciones5['error_absoluto_edkp2'] = abs(estimaciones5['confianza_real'] - estimaciones5['confianza_edkp2'])
estimaciones6['error_absoluto_edkp2'] = abs(estimaciones6['confianza_real'] - estimaciones6['confianza_edkp2'])
estimaciones7['error_absoluto_edkp2'] = abs(estimaciones7['confianza_real'] - estimaciones7['confianza_edkp2'])
estimaciones8['error_absoluto_edkp2'] = abs(estimaciones8['confianza_real'] - estimaciones8['confianza_edkp2'])
estimaciones9['error_absoluto_edkp2'] = abs(estimaciones9['confianza_real'] - estimaciones9['confianza_edkp2'])
estimaciones10['error_absoluto_edkp2'] = abs(estimaciones10['confianza_real'] - estimaciones10['confianza_edkp2'])

# Resultados
eje_x = ['3-16', '17-29', '30-45', '46-65', '66-86', '87-112', '113-154', '155-195', '196-251', '252-422']
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
result.loc[5, 'error_absoluto_prom'] = sum(estimaciones6['error_absoluto_prom']) / len(estimaciones6)
result.loc[6, 'error_absoluto_prom'] = sum(estimaciones7['error_absoluto_prom']) / len(estimaciones7)
result.loc[7, 'error_absoluto_prom'] = sum(estimaciones8['error_absoluto_prom']) / len(estimaciones8)
result.loc[8, 'error_absoluto_prom'] = sum(estimaciones9['error_absoluto_prom']) / len(estimaciones9)
result.loc[9, 'error_absoluto_prom'] = sum(estimaciones10['error_absoluto_prom']) / len(estimaciones10)

result.loc[0, 'error_absoluto_tt'] = sum(estimaciones1['error_absoluto_tt']) / len(estimaciones1)
result.loc[1, 'error_absoluto_tt'] = sum(estimaciones2['error_absoluto_tt']) / len(estimaciones2)
result.loc[2, 'error_absoluto_tt'] = sum(estimaciones3['error_absoluto_tt']) / len(estimaciones3)
result.loc[3, 'error_absoluto_tt'] = sum(estimaciones4['error_absoluto_tt']) / len(estimaciones4)
result.loc[4, 'error_absoluto_tt'] = sum(estimaciones5['error_absoluto_tt']) / len(estimaciones5)
result.loc[5, 'error_absoluto_tt'] = sum(estimaciones6['error_absoluto_tt']) / len(estimaciones6)
result.loc[6, 'error_absoluto_tt'] = sum(estimaciones7['error_absoluto_tt']) / len(estimaciones7)
result.loc[7, 'error_absoluto_tt'] = sum(estimaciones8['error_absoluto_tt']) / len(estimaciones8)
result.loc[8, 'error_absoluto_tt'] = sum(estimaciones9['error_absoluto_tt']) / len(estimaciones9)
result.loc[9, 'error_absoluto_tt'] = sum(estimaciones10['error_absoluto_tt']) / len(estimaciones10)

result.loc[0, 'error_absoluto_edkp1'] = sum(estimaciones1['error_absoluto_edkp1']) / len(estimaciones1)
result.loc[1, 'error_absoluto_edkp1'] = sum(estimaciones2['error_absoluto_edkp1']) / len(estimaciones2)
result.loc[2, 'error_absoluto_edkp1'] = sum(estimaciones3['error_absoluto_edkp1']) / len(estimaciones3)
result.loc[3, 'error_absoluto_edkp1'] = sum(estimaciones4['error_absoluto_edkp1']) / len(estimaciones4)
result.loc[4, 'error_absoluto_edkp1'] = sum(estimaciones5['error_absoluto_edkp1']) / len(estimaciones5)
result.loc[5, 'error_absoluto_edkp1'] = sum(estimaciones6['error_absoluto_edkp1']) / len(estimaciones6)
result.loc[6, 'error_absoluto_edkp1'] = sum(estimaciones7['error_absoluto_edkp1']) / len(estimaciones7)
result.loc[7, 'error_absoluto_edkp1'] = sum(estimaciones8['error_absoluto_edkp1']) / len(estimaciones8)
result.loc[8, 'error_absoluto_edkp1'] = sum(estimaciones9['error_absoluto_edkp1']) / len(estimaciones9)
result.loc[9, 'error_absoluto_edkp1'] = sum(estimaciones10['error_absoluto_edkp1']) / len(estimaciones10)

result.loc[0, 'error_absoluto_edkp2'] = sum(estimaciones1['error_absoluto_edkp2']) / len(estimaciones1)
result.loc[1, 'error_absoluto_edkp2'] = sum(estimaciones2['error_absoluto_edkp2']) / len(estimaciones2)
result.loc[2, 'error_absoluto_edkp2'] = sum(estimaciones3['error_absoluto_edkp2']) / len(estimaciones3)
result.loc[3, 'error_absoluto_edkp2'] = sum(estimaciones4['error_absoluto_edkp2']) / len(estimaciones4)
result.loc[4, 'error_absoluto_edkp2'] = sum(estimaciones5['error_absoluto_edkp2']) / len(estimaciones5)
result.loc[5, 'error_absoluto_edkp2'] = sum(estimaciones6['error_absoluto_edkp2']) / len(estimaciones6)
result.loc[6, 'error_absoluto_edkp2'] = sum(estimaciones7['error_absoluto_edkp2']) / len(estimaciones7)
result.loc[7, 'error_absoluto_edkp2'] = sum(estimaciones8['error_absoluto_edkp2']) / len(estimaciones8)
result.loc[8, 'error_absoluto_edkp2'] = sum(estimaciones9['error_absoluto_edkp2']) / len(estimaciones9)
result.loc[9, 'error_absoluto_edkp2'] = sum(estimaciones10['error_absoluto_edkp2']) / len(estimaciones10)

print(result)

x = result['eje_x']
y1 = result['error_absoluto_prom']
y2 = result['error_absoluto_tt']
y3 = result['error_absoluto_edkp1']
y4 = result['error_absoluto_edkp2']

plt.plot(x, y1, marker='o', label='Primeros Vecinos')
plt.plot(x, y2, marker='o', label='TidalTrust')
plt.plot(x, y3, marker='o', label='GFTrust, ganancia constante')
plt.plot(x, y4, marker='o', label='GFTrust, ganancia decreciente')

plt.xlabel('Número de aristas')
plt.ylabel('Error absoluto medio')
plt.title('Error Absoluto Medio por número de aristas en la WoT')

plt.legend()
plt.show()


# Resultados
eje_x = ['3-16', '17-29', '30-45', '46-65', '66-86', '87-112', '113-154', '155-195', '196-251', '252-422']
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
result.loc[5, 'rmse_prom'] = fn.RMSE(len(estimaciones6), estimaciones6['confianza_prom'], estimaciones6['confianza_real'])
result.loc[6, 'rmse_prom'] = fn.RMSE(len(estimaciones7), estimaciones7['confianza_prom'], estimaciones7['confianza_real'])
result.loc[7, 'rmse_prom'] = fn.RMSE(len(estimaciones8), estimaciones8['confianza_prom'], estimaciones8['confianza_real'])
result.loc[8, 'rmse_prom'] = fn.RMSE(len(estimaciones9), estimaciones9['confianza_prom'], estimaciones9['confianza_real'])
result.loc[9, 'rmse_prom'] = fn.RMSE(len(estimaciones10), estimaciones10['confianza_prom'], estimaciones10['confianza_real'])

result.loc[0, 'rmse_tt'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_tt'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_tt'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_tt'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_tt'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_tt'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_tt'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_tt'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_tt'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_tt'], estimaciones5['confianza_real'])
result.loc[5, 'rmse_tt'] = fn.RMSE(len(estimaciones6), estimaciones6['confianza_tt'], estimaciones6['confianza_real'])
result.loc[6, 'rmse_tt'] = fn.RMSE(len(estimaciones7), estimaciones7['confianza_tt'], estimaciones7['confianza_real'])
result.loc[7, 'rmse_tt'] = fn.RMSE(len(estimaciones8), estimaciones8['confianza_tt'], estimaciones8['confianza_real'])
result.loc[8, 'rmse_tt'] = fn.RMSE(len(estimaciones9), estimaciones9['confianza_tt'], estimaciones9['confianza_real'])
result.loc[9, 'rmse_tt'] = fn.RMSE(len(estimaciones10), estimaciones10['confianza_tt'], estimaciones10['confianza_real'])

result.loc[0, 'rmse_edkp1'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_edkp1'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_edkp1'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_edkp1'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_edkp1'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_edkp1'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_edkp1'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_edkp1'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_edkp1'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_edkp1'], estimaciones5['confianza_real'])
result.loc[5, 'rmse_edkp1'] = fn.RMSE(len(estimaciones6), estimaciones6['confianza_edkp1'], estimaciones6['confianza_real'])
result.loc[6, 'rmse_edkp1'] = fn.RMSE(len(estimaciones7), estimaciones7['confianza_edkp1'], estimaciones7['confianza_real'])
result.loc[7, 'rmse_edkp1'] = fn.RMSE(len(estimaciones8), estimaciones8['confianza_edkp1'], estimaciones8['confianza_real'])
result.loc[8, 'rmse_edkp1'] = fn.RMSE(len(estimaciones9), estimaciones9['confianza_edkp1'], estimaciones9['confianza_real'])
result.loc[9, 'rmse_edkp1'] = fn.RMSE(len(estimaciones10), estimaciones10['confianza_edkp1'], estimaciones10['confianza_real'])

result.loc[0, 'rmse_edkp2'] = fn.RMSE(len(estimaciones1), estimaciones1['confianza_edkp2'], estimaciones1['confianza_real'])
result.loc[1, 'rmse_edkp2'] = fn.RMSE(len(estimaciones2), estimaciones2['confianza_edkp2'], estimaciones2['confianza_real'])
result.loc[2, 'rmse_edkp2'] = fn.RMSE(len(estimaciones3), estimaciones3['confianza_edkp2'], estimaciones3['confianza_real'])
result.loc[3, 'rmse_edkp2'] = fn.RMSE(len(estimaciones4), estimaciones4['confianza_edkp2'], estimaciones4['confianza_real'])
result.loc[4, 'rmse_edkp2'] = fn.RMSE(len(estimaciones5), estimaciones5['confianza_edkp2'], estimaciones5['confianza_real'])
result.loc[5, 'rmse_edkp2'] = fn.RMSE(len(estimaciones6), estimaciones6['confianza_edkp2'], estimaciones6['confianza_real'])
result.loc[6, 'rmse_edkp2'] = fn.RMSE(len(estimaciones7), estimaciones7['confianza_edkp2'], estimaciones7['confianza_real'])
result.loc[7, 'rmse_edkp2'] = fn.RMSE(len(estimaciones8), estimaciones8['confianza_edkp2'], estimaciones8['confianza_real'])
result.loc[8, 'rmse_edkp2'] = fn.RMSE(len(estimaciones9), estimaciones9['confianza_edkp2'], estimaciones9['confianza_real'])
result.loc[9, 'rmse_edkp2'] = fn.RMSE(len(estimaciones10), estimaciones10['confianza_edkp2'], estimaciones10['confianza_real'])

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

plt.xlabel('Número de aristas')
plt.ylabel('Raíz del error cuadrático medio')
plt.title('Raíz del error cuadrático medio por número de aristas en la WoT')

plt.legend()
plt.show()

# Error de estimación
# Error PROM
estimaciones1['error_estimacion_prom'] = estimaciones1['confianza_real'] - estimaciones1['confianza_prom']
estimaciones2['error_estimacion_prom'] = estimaciones2['confianza_real'] - estimaciones2['confianza_prom']
estimaciones3['error_estimacion_prom'] = estimaciones3['confianza_real'] - estimaciones3['confianza_prom']
estimaciones4['error_estimacion_prom'] = estimaciones4['confianza_real'] - estimaciones4['confianza_prom']
estimaciones5['error_estimacion_prom'] = estimaciones5['confianza_real'] - estimaciones5['confianza_prom']
estimaciones6['error_estimacion_prom'] = estimaciones6['confianza_real'] - estimaciones6['confianza_prom']
estimaciones7['error_estimacion_prom'] = estimaciones7['confianza_real'] - estimaciones7['confianza_prom']
estimaciones8['error_estimacion_prom'] = estimaciones8['confianza_real'] - estimaciones8['confianza_prom']
estimaciones9['error_estimacion_prom'] = estimaciones9['confianza_real'] - estimaciones9['confianza_prom']
estimaciones10['error_estimacion_prom'] = estimaciones10['confianza_real'] - estimaciones10['confianza_prom']

# Error TT
estimaciones1['error_estimacion_tt'] = estimaciones1['confianza_real'] - estimaciones1['confianza_tt']
estimaciones2['error_estimacion_tt'] = estimaciones2['confianza_real'] - estimaciones2['confianza_tt']
estimaciones3['error_estimacion_tt'] = estimaciones3['confianza_real'] - estimaciones3['confianza_tt']
estimaciones4['error_estimacion_tt'] = estimaciones4['confianza_real'] - estimaciones4['confianza_tt']
estimaciones5['error_estimacion_tt'] = estimaciones5['confianza_real'] - estimaciones5['confianza_tt']
estimaciones6['error_estimacion_tt'] = estimaciones6['confianza_real'] - estimaciones6['confianza_tt']
estimaciones7['error_estimacion_tt'] = estimaciones7['confianza_real'] - estimaciones7['confianza_tt']
estimaciones8['error_estimacion_tt'] = estimaciones8['confianza_real'] - estimaciones8['confianza_tt']
estimaciones9['error_estimacion_tt'] = estimaciones9['confianza_real'] - estimaciones9['confianza_tt']
estimaciones10['error_estimacion_tt'] = estimaciones10['confianza_real'] - estimaciones10['confianza_tt']

# Error EDK1
estimaciones1['error_estimacion_edkp1'] = estimaciones1['confianza_real'] - estimaciones1['confianza_edkp1']
estimaciones2['error_estimacion_edkp1'] = estimaciones2['confianza_real'] - estimaciones2['confianza_edkp1']
estimaciones3['error_estimacion_edkp1'] = estimaciones3['confianza_real'] - estimaciones3['confianza_edkp1']
estimaciones4['error_estimacion_edkp1'] = estimaciones4['confianza_real'] - estimaciones4['confianza_edkp1']
estimaciones5['error_estimacion_edkp1'] = estimaciones5['confianza_real'] - estimaciones5['confianza_edkp1']
estimaciones6['error_estimacion_edkp1'] = estimaciones6['confianza_real'] - estimaciones6['confianza_edkp1']
estimaciones7['error_estimacion_edkp1'] = estimaciones7['confianza_real'] - estimaciones7['confianza_edkp1']
estimaciones8['error_estimacion_edkp1'] = estimaciones8['confianza_real'] - estimaciones8['confianza_edkp1']
estimaciones9['error_estimacion_edkp1'] = estimaciones9['confianza_real'] - estimaciones9['confianza_edkp1']
estimaciones10['error_estimacion_edkp1'] = estimaciones10['confianza_real'] - estimaciones10['confianza_edkp1']

# Error EDKP2
estimaciones1['error_estimacion_edkp2'] = estimaciones1['confianza_real'] - estimaciones1['confianza_edkp2']
estimaciones2['error_estimacion_edkp2'] = estimaciones2['confianza_real'] - estimaciones2['confianza_edkp2']
estimaciones3['error_estimacion_edkp2'] = estimaciones3['confianza_real'] - estimaciones3['confianza_edkp2']
estimaciones4['error_estimacion_edkp2'] = estimaciones4['confianza_real'] - estimaciones4['confianza_edkp2']
estimaciones5['error_estimacion_edkp2'] = estimaciones5['confianza_real'] - estimaciones5['confianza_edkp2']
estimaciones6['error_estimacion_edkp2'] = estimaciones6['confianza_real'] - estimaciones6['confianza_edkp2']
estimaciones7['error_estimacion_edkp2'] = estimaciones7['confianza_real'] - estimaciones7['confianza_edkp2']
estimaciones8['error_estimacion_edkp2'] = estimaciones8['confianza_real'] - estimaciones8['confianza_edkp2']
estimaciones9['error_estimacion_edkp2'] = estimaciones9['confianza_real'] - estimaciones9['confianza_edkp2']
estimaciones10['error_estimacion_edkp2'] = estimaciones10['confianza_real'] - estimaciones10['confianza_edkp2']

medianas =[]
medias = []

# Crear el diagrama de cajas
data = [estimaciones1['error_estimacion_prom'], estimaciones1['error_estimacion_tt'], estimaciones1['error_estimacion_edkp1'], estimaciones1['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True,boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 3-16 aristas')
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
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 17-29 aristas')
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
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 30-45 aristas')
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
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 46-65 aristas')
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
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 66-86 aristas')
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

data = [estimaciones6['error_estimacion_prom'], estimaciones6['error_estimacion_tt'], estimaciones6['error_estimacion_edkp1'], estimaciones6['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 87-112 aristas')
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

data = [estimaciones7['error_estimacion_prom'], estimaciones7['error_estimacion_tt'], estimaciones7['error_estimacion_edkp1'], estimaciones7['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 113-154 aristas')
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

data = [estimaciones8['error_estimacion_prom'], estimaciones8['error_estimacion_tt'], estimaciones8['error_estimacion_edkp1'], estimaciones8['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 155-195 aristas')
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

data = [estimaciones9['error_estimacion_prom'], estimaciones9['error_estimacion_tt'], estimaciones9['error_estimacion_edkp1'], estimaciones9['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 196-251 aristas')
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

data = [estimaciones10['error_estimacion_prom'], estimaciones10['error_estimacion_tt'], estimaciones10['error_estimacion_edkp1'], estimaciones10['error_estimacion_edkp2']]
plt.boxplot(data, vert=True, patch_artist=True, boxprops = dict(facecolor = "lightblue"))
plt.xticks([1, 2, 3, 4], ['Primeros \nVecinos', 'TidalTrust', 'GFTrust, \nconstante', 'GFTrust, \ndecreciente'])
plt.ylabel('Error de la estimación')
plt.title('WoT con 252-422 aristas')
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

medianas_res = pd.DataFrame(medianas, columns=['Primeros Vecinos', 'TidalTrust', 'GFTrust constante', 'GFTrust decreciente'])
medias_res = pd.DataFrame(medias, columns=['Primeros Vecinos', 'TidalTrust', 'GFTrust constante', 'GFTrust decreciente'])

print(medianas_res)
print(medias_res)