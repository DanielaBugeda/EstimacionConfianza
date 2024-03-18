import pandas as pd
import networkx as nx
from datetime import datetime

BBDD = pd.read_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/BBDD_Bitcoin_original.txt')

# Crear una digráfica usando el paquete networkx
red = nx.MultiDiGraph()

# Nodos
nodos = set(list(BBDD['source']) + list(BBDD['target']))
red.add_nodes_from(nodos)

# Aristas
for i, fila in BBDD.iterrows():
    nodoCabeza = fila['source']  
    nodoCola = fila['target']  
    confianza = fila['trust']
    fecha = datetime.fromtimestamp(fila['time']).strftime('%Y-%m-%d %H:%M:%S UTC')
    red.add_edge(nodoCabeza, nodoCola, confianza=confianza, fecha=fecha)
  
print(red.number_of_edges())
print(red.number_of_nodes())

# Filtrar aristas duplicadas (declaraciones de confianza viejas)
for nodoA in nodos:
    for nodoB in nodos:
        list_edges_X = list()
        x = red.number_of_edges(nodoA, nodoB)
        
        if x > 1:
            data_dict = red.get_edge_data(nodoA, nodoB)
            key_with_max_fecha = max(data_dict.keys(), key=lambda k: data_dict[k]['fecha'])
            l = list(range(x))
            l.remove(key_with_max_fecha)
            for i in l:
                red.remove_edge(nodoA, nodoB, i)

print(red.number_of_edges())
print(red.number_of_nodes())

# Normalizar valores de confianza
aristas_info = []
for a in red.edges(data=True):
    source, target, attributes = a
    trust = attributes['confianza']
    aristas_info.append((source, target, trust))
    
red_df = pd.DataFrame(aristas_info, columns=['source', 'target', 'trust'])
max_trust = red_df['trust'].max()
min_trust = red_df['trust'].min()

red_df['trust'] = red_df['trust'].apply(lambda x: (x-min_trust)/(max_trust-min_trust))
red_df.to_csv('/Users/danielabugeda/Documents/ITAM/Tesis/Códigos Octubre/Versiones Finales/BBDD_Bitcoin.txt', header=True, index=False, sep=',')