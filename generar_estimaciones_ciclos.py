import funciones as fn
import pandas as pd
from datetime import datetime
import time

dataBitCoin = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/BBDD_Bitcoin.txt')
dataSubgraficas = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/archivo_datos_con_ciclos.txt')
resultados = []

for i, fila in dataSubgraficas.iterrows():
    print(i)
    
    current_timestamp = datetime.now().timestamp()
    dt_object = datetime.utcfromtimestamp(current_timestamp)
    formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')
    print("Human-Readable Date:", formatted_date)

    archivo = fila['archivo']
    fuente = fila['fuente']
    sumidero = fila['sumidero']
    
    df_archivo = pd.read_csv(archivo)

    # Generar red original y guardar datos 
    red_original = fn.generarRed(df_archivo)
    subred_nodos = red_original.number_of_nodes()
    subred_aristas = red_original.number_of_edges()
    
    # Generar Web of Trust y guardar datos 
    st_WoT = time.perf_counter()
    WoT = fn.crearWoT(red_original, fuente, sumidero) 
    WoT_nodos = WoT.number_of_nodes()
    WoT_aristas = WoT.number_of_edges()
    et_WoT = time.perf_counter()
    execution_time_WoT = st_WoT - et_WoT
    
    # Guardar la confianza real
    confianza_BBDD = dataBitCoin[(dataBitCoin['source'] == fuente) & (dataBitCoin['target'] == sumidero)]['trust']
    confianza_BBDD = confianza_BBDD.reset_index()
    confianza_real = confianza_BBDD['trust'].iloc[0]  
    WoT.remove_edge(fuente, sumidero)
    
    # Calcular distancia fuente-sumidero
    dist = fn.distanciaNodos(WoT)
    distFS = dist[fuente][sumidero]
    
    # Calcular confianza promedio por conexiones directas para comparación
    conexiones_sumidero = [(u, v) for u, v in WoT.edges() if v == sumidero]
    n = len(conexiones_sumidero)
    confianza_prom = 0
    for e in conexiones_sumidero:
        confianza_prom += WoT.get_edge_data(*e)['confianza']
    confianza_prom = confianza_prom / n

    # Edmonds Karp con delta constante
    st_edkp1 = time.perf_counter()
    delta1 = {0: 0, 1: .90, 2: .90, 3: .90, 4: .90, 5:.90, 6:.90, 7: .90, 8: .90, 9: .90, 10: .90}
    st_redGen = time.perf_counter()
    red_generalizada = fn.crearRedGeneralizada(WoT, fuente, sumidero, delta1)
    redGen_nodos = red_generalizada.number_of_nodes()
    redGen_aristas = red_generalizada.number_of_edges()
    et_redGen = time.perf_counter()
    execution_time_redGen = st_redGen - et_redGen
    confianza_edkp1 = fn.EdmondsKarp(red_generalizada, fuente, sumidero)
    et_edkp1 = time.perf_counter()
    execution_time_edkp1 = st_edkp1 - et_edkp1
    
    # Edmonds Karp con delta decreciente (advogato)
    st_edkp2 = time.perf_counter()
    delta2 = fn.advogato(WoT,fuente, sumidero)
    red_generalizada = fn.crearRedGeneralizada(WoT, fuente, sumidero, delta2)
    confianza_edkp2 = fn.EdmondsKarp(red_generalizada, fuente, sumidero)
    et_edkp2 = time.perf_counter()
    execution_time_edkp2 = st_edkp2 - et_edkp2
    
    # Guardar estimación
    result = { 
        'subred_nodos' : subred_nodos,
        'subred_aristas' : subred_aristas,
        'WoT_nodos' : WoT_nodos,
        'WoT_aristas' : WoT_aristas,
        'tiempo_Wot' : execution_time_WoT,
        'distFS' : distFS,
        'redGen_nodos' : redGen_nodos,
        'redGen_aristas' : redGen_aristas,
        'tiempo_redGen' : execution_time_redGen,
        'confianza_real' : confianza_real,
        'confianza_prom' : confianza_prom,
        'confianza_edkp1' : confianza_edkp1,
        'confianza_edkp2' : confianza_edkp2,
        'tiempo_edkp1' : execution_time_edkp1,
        'tiempo_edkp2' : execution_time_edkp2
    }
    resultados.append(result)
    print(result)

resultados = pd.DataFrame(resultados)
resultados.to_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/estimaciones_ciclos.txt', header=True, index=False, sep=',')