import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Exportar tabla de estimaciones
estimaciones = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/estimaciones.txt', dtype=float)
confianza_real = list ()
confianza_prom = list ()
confianza_tt = list ()
confianza_edkp1 = list ()
confianza_edkp2 = list ()
resultados = []

for i, fila in estimaciones.iterrows():
    confianza_real.append(fila['confianza_real'])
    confianza_prom.append(fila['confianza_prom'])
    confianza_tt.append(fila['confianza_tt'])
    confianza_edkp1.append(fila['confianza_edkp1'])
    confianza_edkp2.append(fila['confianza_edkp2'])

def categorize_result(result_data, result_pred):
    if (result_data == True) & (result_pred == True):
        return 'TP'
    elif (result_data == True) & (result_pred == False):
        return 'FN'
    elif (result_data == False) & (result_pred == True):
        return 'FP'
    elif (result_data == False) & (result_pred == False):
        return 'TN'


def model_metrics(clasif):
    TP = clasif.count('TP')
    TN = clasif.count('TN')
    FP = clasif.count('FP')
    FN = clasif.count('FN')
    
    if (TP + TN + FP + FN) != 0:
        Exac = (TP + TN) / (TP + TN + FP + FN)
    else:
        Exac = 0
    
    if (TP + FP) != 0:
        Precc = TP / (TP + FP)
    else:
        Precc = 0
        
    if (TP + FN) != 0:
        Recall = TP / (TP + FN)
    else:
        Recall = 0
        
    if (FP + TN) != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 0
        
    if (Precc + Recall) != 0:
        F1 = (2*Precc*Recall) / (Precc + Recall)
    else:
            F1 = 0
        
    return Exac, Precc, Recall, FPR, F1

th =  [estimaciones['confianza_prom'].mean(), 0.5, 0.6, 0.7, 0.8]

for i in th:
    estimaciones['confiable_Data_real'] = estimaciones['confianza_real'] >= i
    estimaciones['confiable_Data_prom'] = estimaciones['confianza_prom'] >= i
    estimaciones['confiable_Pred_tt'] = estimaciones['confianza_tt'] >= i
    estimaciones['confiable_Pred_edkp1'] = estimaciones['confianza_edkp1'] >= i
    estimaciones['confiable_Pred_edkp2'] = estimaciones['confianza_edkp2'] >= i

    estimaciones['clasif_PROM_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Data_prom']), axis=1)
    estimaciones['clasif_TT_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Pred_tt']), axis=1)
    estimaciones['clasif_EDKP1_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Pred_edkp1']), axis=1)
    estimaciones['clasif_EDKP2_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Pred_edkp2']), axis=1)

    model_metrics_PROM_real = model_metrics(estimaciones['clasif_PROM_real'].tolist())
    model_metrics_TT_real = model_metrics(estimaciones['clasif_TT_real'].tolist())
    model_metrics_edkp1_real = model_metrics(estimaciones['clasif_EDKP1_real'].tolist())
    model_metrics_edkp2_real = model_metrics(estimaciones['clasif_EDKP2_real'].tolist())
    
    # Guardar RESULTADOS
    result = { 
        'th' : i,
        'Exac_PROM' : model_metrics_PROM_real[0],
        'Prec_PROM' : model_metrics_PROM_real[1],
        'Rec_PROM' : model_metrics_PROM_real[2],
        'FPR_PROM' : model_metrics_PROM_real[3],
        'F1_PROM' : model_metrics_PROM_real[4],
        'Exac_TT' : model_metrics_TT_real[0],
        'Prec_TT' : model_metrics_TT_real[1],
        'Rec_TT' : model_metrics_TT_real[2],
        'FPR_TT' : model_metrics_TT_real[3],
        'F1_TT' : model_metrics_TT_real[4],
        'Exac_EDKP1' : model_metrics_edkp1_real[0],
        'Prec_EDKP1' : model_metrics_edkp1_real[1],
        'Rec_EDKP1' : model_metrics_edkp1_real[2],
        'FPR_EDKP1' : model_metrics_edkp1_real[3],
        'F1_EDKP1' : model_metrics_edkp1_real[4],
        'Exac_EDKP2' : model_metrics_edkp2_real[0],
        'Prec_EDKP2' : model_metrics_edkp2_real[1],
        'Rec_EDKP2' : model_metrics_edkp2_real[2],
        'FPR_EDKP2' : model_metrics_edkp2_real[3],
        'F1_EDKP2' : model_metrics_edkp2_real[4],
    }
    resultados.append(result)
    
resultados = pd.DataFrame(resultados)
print(resultados)

resultados.to_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/metricas.txt', header=True, index=False, sep=',')


resultados = resultados.sort_values('th')
x = resultados['th']
y1 = resultados['F1_PROM']
y2 = resultados['F1_TT']
y3 = resultados['F1_EDKP1']
y4 = resultados['F1_EDKP2']

plt.plot(x, y1, marker='o', label='Primeros Vecinos')
plt.plot(x, y2, marker='o', label='TidalTrust')
plt.plot(x, y3, marker='o', label='GFTrust, ganancia constante')
plt.plot(x, y4, marker='o', label='GFTrust, ganancia decreciente')

plt.xlabel('Umbral de confianza')
plt.ylabel('F1-Sore')
plt.title('Tendencia del F1-Score en función del umbral de decisión')

plt.legend()
plt.show()



# Curva ROC
th = np.linspace(0.1,0.9,40)
resultados = []
for i in th:
    estimaciones['confiable_Data_real'] = estimaciones['confianza_real'] >= i
    estimaciones['confiable_Data_prom'] = estimaciones['confianza_prom'] >= i
    estimaciones['confiable_Pred_tt'] = estimaciones['confianza_tt'] >= i
    estimaciones['confiable_Pred_edkp1'] = estimaciones['confianza_edkp1'] >= i
    estimaciones['confiable_Pred_edkp2'] = estimaciones['confianza_edkp2'] >= i

    estimaciones['clasif_PROM_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Data_prom']), axis=1)
    estimaciones['clasif_TT_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Pred_tt']), axis=1)
    estimaciones['clasif_EDKP1_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Pred_edkp1']), axis=1)
    estimaciones['clasif_EDKP2_real'] = estimaciones.apply(lambda row: categorize_result(row['confiable_Data_real'], row['confiable_Pred_edkp2']), axis=1)

    model_metrics_PROM_real = model_metrics(estimaciones['clasif_PROM_real'].tolist())
    model_metrics_TT_real = model_metrics(estimaciones['clasif_TT_real'].tolist())
    model_metrics_edkp1_real = model_metrics(estimaciones['clasif_EDKP1_real'].tolist())
    model_metrics_edkp2_real = model_metrics(estimaciones['clasif_EDKP2_real'].tolist())
    
    # Guardar RESULTADOS
    result = { 
        'th' : i,
        'Rec_PROM' : model_metrics_PROM_real[2],
        'FPR_PROM' : model_metrics_PROM_real[3],
        'Rec_TT' : model_metrics_TT_real[2],
        'FPR_TT' : model_metrics_TT_real[3],
        'Rec_EDKP1' : model_metrics_edkp1_real[2],
        'FPR_EDKP1' : model_metrics_edkp1_real[3],
        'Rec_EDKP2' : model_metrics_edkp2_real[2],
        'FPR_EDKP2' : model_metrics_edkp2_real[3]
    }
    resultados.append(result)
    
resultados = pd.DataFrame(resultados)
print(resultados)
resultados.to_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/ROC.txt', header=True, index=False, sep=',')


resultados = resultados.sort_values('FPR_PROM')
x = resultados['FPR_PROM']
y = resultados['Rec_PROM']
auc = metrics.auc(x,y)
plt.plot(x, y, color='#1f77b4')
plt.fill_between(x,y, color='#1f77b4', alpha=0.4)
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para la heurística Primeros Vecinos')
text = 'AUC = ' + str(round(auc, 2))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, text, fontsize=12,
        verticalalignment='top', bbox=props)
plt.show()

resultados = resultados.sort_values('FPR_TT')
x = resultados['FPR_TT']
y = resultados['Rec_TT']
auc = metrics.auc(x,y)
plt.plot(x, y, color='#ff7f0e')
plt.fill_between(x,y, color='#ff7f0e', alpha=0.4)
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para el algoritmo TidalTrust')
text = 'AUC = ' + str(round(auc, 2))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, text, fontsize=12,
        verticalalignment='top', bbox=props)
plt.show()

resultados = resultados.sort_values('FPR_EDKP1')
x = resultados['FPR_EDKP1']
y = resultados['Rec_EDKP1']
auc = metrics.auc(x,y)
plt.plot(x, y, color='#2ca02c')
plt.fill_between(x,y, color='#2ca02c', alpha=0.4)
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para el algoritmo GFTrust Constante')
text = 'AUC = ' + str(round(auc, 2))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, text, fontsize=12,
        verticalalignment='top', bbox=props)
plt.show()

resultados = resultados.sort_values('FPR_EDKP2')
x = resultados['FPR_EDKP2']
y = resultados['Rec_EDKP2']
auc = metrics.auc(x,y)
plt.plot(x, y, color='#d62728')
plt.fill_between(x,y, color='#d62728', alpha=0.4)
plt.xlim(0,1)
plt.ylim(0,1.1)
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para el algoritmo GFTrust Decreciente')
text = 'AUC = ' + str(round(auc, 2))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, text, fontsize=12,
        verticalalignment='top', bbox=props)
plt.show()