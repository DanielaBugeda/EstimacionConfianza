# Método para generar las estimaciones
# 1. Crear las subgráficas de la data original. Crear 30 subgráficas.
# 2. Crear 50 combinaciones fuente sumidero por cada subgráficas.
# 3. Obtener la confianza de cada fuente-sumidero con ambos algoritmos (TidalTrust y GFTrust)

import funciones as fn
import pandas as pd

# 1. Crear las subgráficas de la data original. Crear 30 subgráficas.
# Lectura de base de datos como red dirigida
dataBitcoin = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/BBDD_Bitcoin.txt', dtype=str)
red = fn.generarRed(dataBitcoin)

variablesSubgraficas = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/variables_subgraficas.txt', dtype=str)

archivo_datos_sin_ciclos = pd.DataFrame()
archivo_datos_con_ciclos = pd.DataFrame()
for i, fila in variablesSubgraficas.iterrows():
    print(i)
    fuente = fila['fuente']
    n = int(fila['n'])
    x = int(fila['x'])
    
    # Generar árbol y subgráfica
    arbol = fn.tree(red, fuente, n, x)
    subred = fn.subraph(red, arbol)
    
    subred_ciclos = fn.subraph_cycles(red, arbol)

    # Obtener información de aristas red sin ciclos
    aristas_info = []
    for a in subred.edges(data=True):
        source, target, attributes = a
        trust = attributes['confianza']
        aristas_info.append((source, target, trust))

    # Crear archivo con información de la subred sin ciclos
    subred_df = pd.DataFrame(aristas_info, columns=['source', 'target', 'trust'])
    archivo1 = '/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/Subgraficas/Subgrafica' + str(i+1) + '.txt'
    subred_df.to_csv(archivo1, header=True, index=False, sep=',')
    
    # Obtener información de aristas red con ciclos
    aristas_info = []
    for a in subred_ciclos.edges(data=True):
        source, target, attributes = a
        trust = attributes['confianza']
        aristas_info.append((source, target, trust))

    # Crear archivo con información de la subred con ciclos
    subred_ciclos_df = pd.DataFrame(aristas_info, columns=['source', 'target', 'trust'])
    archivo2 = '/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/Subgraficas/Subgrafica_ciclos_' + str(i+1) + '.txt'
    subred_ciclos_df.to_csv(archivo2, header=True, index=False, sep=',')
    
    # 2. Crear y combinaciones fuente sumidero por cada subgráficas.
    nodos = set(subred.nodes)
    nodos.remove(fuente)
    
    sumideros = set(dataBitcoin[(dataBitcoin['source'] == fuente)]['target'])
    if fuente in sumideros:
        sumideros.remove(fuente)

    sumideros = list(sumideros.intersection(nodos)) 
    
    dist = fn.distanciaNodos(arbol)[fuente]
    sumideros_final = []
    for s in sumideros:
        if dist[s] >= 2:
            sumideros_final.append(s)
            
    y = len(sumideros_final)
    
    # Archivos sin ciclos
    df1 = pd.DataFrame({'archivo': [archivo1] * y, 'fuente': [fuente] * y})
    df1['sumidero'] = sumideros_final
           
    archivo_datos_sin_ciclos = pd.concat([archivo_datos_sin_ciclos, df1], ignore_index=True)
    
    # Archivos con ciclos
    df2 = pd.DataFrame({'archivo': [archivo2] * y, 'fuente': [fuente] * y})
    df2['sumidero'] = sumideros_final
           
    archivo_datos_con_ciclos = pd.concat([archivo_datos_con_ciclos, df2], ignore_index=True)
    
archivo_datos_sin_ciclos.to_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/archivo_datos_sin_ciclos.txt', header=True, index=False, sep=',')
archivo_datos_con_ciclos.to_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/archivo_datos_con_ciclos.txt', header=True, index=False, sep=',')
