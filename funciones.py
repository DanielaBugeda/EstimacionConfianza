# Librerías utilizadas
import math
from itertools import islice
import networkx as nx

# Función para generar el árbol 
def tree(red, fuente, n, x):
# Método para sacar subgráficas (árboles):
# 1. Escoger aleatoriamente un nodo fuente.
# 2. Establecer un número n de niveles para la subgráfica - ramas hacia adelante del árbol.
# 3. Tomar los x vecinos de mayor grado (número de vecinos).
# 4. Para cada nodo seleccionado, repetir el paso 3.
# 5. Revisar que se incluyan todas las conexiones de los nodos participantes.
    queue = set()
    visitados = set()
    arbol = nx.DiGraph()
    
    queue.add(fuente)
    visitados.add(fuente)
    arbol.add_node(fuente)
    
    arbol.nodes[fuente]['nivel'] = 1

    while queue:
        nodo = next(iter(queue))
        queue.discard(nodo)
            
        degree = dict(red.degree())

        vecinos = list(red.neighbors(nodo))
        vecinos_grados = {v: None for v in vecinos}

        for v in vecinos:
            vecinos_grados[v] = degree[v]
        for q in visitados:
            if q in vecinos_grados.keys():
                del vecinos_grados[q]
        vecinos_ordenado = dict(islice(sorted(vecinos_grados.items(), key=lambda item: item[1], reverse=True),x))
        
        for v in vecinos_ordenado:
            arbol.add_node(v)
            if v not in visitados:
                arbol.add_edge(nodo, v)
                visitados.add(v)
            if (len(nx.shortest_path(arbol, fuente, v))-1) < n:
                queue.add(v)
            arbol.nodes[v]['nivel'] = len(nx.shortest_path(arbol, fuente, v))
            
    return arbol

# Función para agregar aristas hacia adelante sin ciclos
def subraph(red, arbol):
    subred = nx.DiGraph()
    nodos = arbol.nodes()
    
    subred.add_nodes_from(nodos)
    
    for nodoA in arbol.nodes():
        for nodoB in arbol.nodes():
            if (arbol.nodes[nodoB]['nivel'] >= arbol.nodes[nodoA]['nivel']) & (red.has_edge(nodoA, nodoB)) & (not(subred.has_edge(nodoB, nodoA))):
                confianza = red[nodoA][nodoB]['confianza']
                subred.add_edge(nodoA, nodoB, confianza=confianza)
                if list(nx.simple_cycles(subred)):
                    subred.remove_edge(nodoA, nodoB)
                
    return subred

# Función para agregar aristas hacia adelante con ciclos
def subraph_cycles(red, arbol):
    subred = nx.DiGraph()
    nodos = arbol.nodes()
    
    subred.add_nodes_from(nodos)
    
    for nodoA in arbol.nodes():
        for nodoB in arbol.nodes():
            if (arbol.nodes[nodoB]['nivel'] >= arbol.nodes[nodoA]['nivel']) & (red.has_edge(nodoA, nodoB)) & (not(subred.has_edge(nodoB, nodoA))):
                confianza = red[nodoA][nodoB]['confianza']
                subred.add_edge(nodoA, nodoB, confianza=confianza)
                
    return subred

def generarRed(data):
    # Crear una digráfica usando el paquete networkx
    red_original = nx.DiGraph()
    
    # Nodos
    nodos = set(list(data['source']) + list(data['target']))
    red_original.add_nodes_from(nodos)

    # Aristas
    for i, fila in data.iterrows():
        nodoCabeza = fila['source']  
        nodoCola = fila['target']  
        confianza = float(fila['trust'])
        red_original.add_edge(nodoCabeza, nodoCola, confianza=confianza)
        
    return red_original

# Algoritmo Depth-First Search para encontrar los caminos entre el nodo fuente y sumidero. Devuelve los caminos que conforman a la WOT.
def dfs_WOT(red, fuente, sumidero, camino=[]):
    
    # Inicilizar el camino con el nodo fuente
    camino = camino + [fuente]
  
    # Recursión: caso base
    if fuente == sumidero:
        return [camino]

    # Recursión: iteraciones
    if fuente not in red:
        return []

    caminos = [] 

    for vecino in red[fuente]:
        if vecino not in camino:
            caminos_nuevos = dfs_WOT(red, vecino, sumidero, camino)
            for camino_nuevo in caminos_nuevos:
                caminos.append(camino_nuevo)

    return caminos # Devuelve todos los posibles caminos entre fuente y sumidero

def crearWoT(red_original, fuente, sumidero):
    # Llamar la función para reducir la red a una WoT
    caminos_WoT = dfs_WOT(red_original, fuente, sumidero)  

    # Crear la WoT en función de los caminos encontrados por DFS
    WoT = nx.DiGraph()
    for camino in caminos_WoT:
        for i in range(len(camino) - 1):
            nodoCabeza = camino[i]
            nodoCola = camino[i + 1]
            arista_data =  red_original.get_edge_data(nodoCabeza, nodoCola)
            confianza = arista_data['confianza']
            WoT.add_edge(nodoCabeza, nodoCola, confianza=confianza)
    
    return WoT

## ALGORITMO TIDAL TRUST
### Función para calcular el nivel máximo de confianza para TT.
def getGraphScores(fuente, sumidero, WoT, scores):
    
    def getNodeScore(curr):
        
        score = 0
        
        # el nodo actual no se encuentra en la gráfica (no debería pasar)
        if curr not in WoT.nodes():
            scores[curr] = 0
            return
        
        # si es nodo que proviene del nodo inicial
        if curr in WoT[fuente]:
            arista_data = WoT.get_edge_data(fuente, curr)
            scores[curr] = arista_data['confianza']
            return
        
        for prev in WoT.predecessors(curr):
            if prev not in scores:
                getNodeScore(prev)
            if curr == sumidero: # si es nodo final
                score = max(score, scores[prev])
            else: # si es nodo intermedio
                arista_data = WoT.get_edge_data(prev, curr)
                score = max(score, min(scores[prev], arista_data['confianza']))
        scores[curr] = score
        
    # empezamos con el sumidero
    getNodeScore(sumidero)
    
### Función para propagar la confianza en TT
def propagateTrust(fuente, sumidero, nivel_confianza, WoT, confianza):
    confianza[sumidero] = 0
    
    def getNodeTrust(curr):
        confianzaNodo = 0
        
        if curr not in WoT.nodes():
            confianza[curr] = 0
            return
        
        for sig in WoT[curr]:
            
            if sig not in confianza:
                getNodeTrust(sig)
            if curr in WoT.predecessors(sumidero):
                arista_data = WoT.get_edge_data(curr, sumidero)
                confianza[curr] = arista_data['confianza'] 
            else:
                sum1 = 0
                sum2 = 0
                arista_data = WoT.get_edge_data(curr, sig)

                if (arista_data['confianza'] < nivel_confianza) & (curr != fuente):
                    confianza[curr] = 0
                elif arista_data['confianza'] >= nivel_confianza:
                    sum1 = sum1+arista_data['confianza']*confianza[sig]
                    sum2 = sum2+arista_data['confianza']
                    
                    if(sum2 != 0):
                        confianzaNodo = sum1/sum2
                        confianza[curr] = confianzaNodo
            
    getNodeTrust(fuente)
    
### Función TidalTrust
def tidalTrust(WoT, fuente, sumidero):
    calificaciones = {}
    confianza = {}
    getGraphScores(fuente, sumidero, WoT, calificaciones)
    nivel_confianza = calificaciones[sumidero]
    propagateTrust(fuente, sumidero, nivel_confianza, WoT, confianza)
    return calificaciones , confianza

def distanciaNodos(red):
    dist = nx.all_pairs_shortest_path_length(red) 
    distancias = {x[0]:x[1] for x in dist}  
    return distancias         

def crearRedGeneralizada(WoT, fuente, sumidero, delta):
    # Generar red generalizada
    red_generalizada = nx.DiGraph()
    nodos_intermedios = set(WoT.nodes())
    nodos_intermedios.remove(fuente)
    nodos_intermedios.remove(sumidero)
    
    # Calcular distancia entre nodos
    dist_Nodos = distanciaNodos(WoT)
    
    # Node Splitting y aristas intermedias
    for nodo in nodos_intermedios:
        nodo_mas = str(nodo) + '+'
        nodo_menos = str(nodo) + '-'
        red_generalizada.add_node(nodo_mas)
        red_generalizada.add_node(nodo_menos)
        delta_nodo = dist_Nodos[fuente][nodo]
        red_generalizada.add_edge(nodo_mas, nodo_menos, ganancia=delta[delta_nodo], capacidad=1)
        
    # Asignación de ganancias y capacidades
    red_generalizada.add_node(fuente)
    red_generalizada.add_node(sumidero) 

    for edge in WoT.edges():
        nodoCabeza = edge[0] 
        nodoCola = edge[1]  
        capacidad = WoT[nodoCabeza][nodoCola]['confianza']
        
        if nodoCabeza == fuente:
            nodo_mas = str(nodoCola) + '+'
            red_generalizada.add_edge(fuente, nodo_mas, ganancia=1, capacidad=capacidad)
        
        elif nodoCola == sumidero:
            nodo_menos = str(nodoCabeza) + '-'
            red_generalizada.add_edge(nodo_menos, sumidero, ganancia=1, capacidad=capacidad)
        
        else:
            nodo_mas = str(nodoCola) + '+'
            nodo_menos = str(nodoCabeza) + '-'
            red_generalizada.add_edge(nodo_menos, nodo_mas, ganancia=1, capacidad=capacidad)
            
    return red_generalizada

## ALGORITMO EDMONDS KARP
def bfs(red_generalizada, fuente, sumidero):
    queue = list()
    nodos_visitados = list()
    predecesor = dict()

    queue.append(fuente)
    nodos_visitados.append(fuente)
    predecesor['-'] = fuente

    while queue:
        nodo = next(iter(queue))
        queue.remove(nodo)

        for vecino in red_generalizada[nodo]:
            capacidad_data = red_generalizada.get_edge_data(nodo, vecino)

            if (vecino not in nodos_visitados) & (capacidad_data['capacidad'] > 0):
                if vecino == sumidero:
                    predecesor[vecino] = nodo
                    return predecesor
                queue.append(vecino)
                nodos_visitados.append(vecino)
                predecesor[vecino] = nodo

def camino_de_aumento(pred, fuente, sumidero):
    camino = list()
    camino.append(sumidero)

    nodo = sumidero
    while nodo != fuente:
        camino.append(pred[nodo])
        nodo = pred[nodo]

    camino.reverse()

    i = len(camino)/2
    j = 0
    camino_edges = list()
    while j <= i:
        arista = (camino[j], camino [j+1])
        camino_edges.append(arista)
        j = j+1
        
    return camino_edges

def EdmondsKarp(red_generalizada, fuente, sumidero):
    flujo_inicial = 1
    flujo_maximo = 0

    while flujo_inicial > 0:
        flujo_delta = flujo_inicial
        pred = bfs(red_generalizada=red_generalizada, fuente=fuente, sumidero=sumidero)
        if pred is None:
            break
        
        camino = camino_de_aumento(pred, fuente, sumidero)
        
        aux = list()
        aux1 = list()
        
        if not camino:
            break

        for arista in camino:
            nodoCabeza = arista[0]
            nodoCola = arista[1]
            arista_data = red_generalizada.get_edge_data(nodoCabeza, nodoCola)
            
            aux.append(flujo_inicial)
            aux.append(arista_data['capacidad'])
            aux.append(arista_data['ganancia']*flujo_delta)
            
            flujo_delta = min(aux)
            aux1.append(flujo_delta)
            
            capacidad_residual = arista_data['capacidad'] - flujo_delta
            red_generalizada[nodoCabeza][nodoCola]['capacidad'] = capacidad_residual
            
            if nodoCabeza == fuente:
                flujo_inicial_aux = flujo_inicial - flujo_delta
            
        flujo_delta = min(aux1)
    
        flujo_maximo = flujo_maximo + flujo_delta
        flujo_inicial = flujo_inicial_aux

    return flujo_maximo

# Root Mean Square Error
# Entre más bajo el valor, más precisión
def RMSE(n, pred, act):
    sum = 0
    for i in zip(pred,act):
        sum += (i[0] - i[1])**2
        
    rmse = math.sqrt(sum/n)
    return rmse

# Pearson Correlation Coefficient
# Entre más cercano a 1, mayor correlación (grado de relación entre la predicción y el valor real)
def PCC(n, pred, act):
    sum1 = 0
    sum2 = 0 
    sum3 = 0
    media_p = 0
    media_a = 0
    
    #Medias muestrales
    for i in zip(pred,act):
        media_p += i[0]
        media_a += i[1]
    media_p = media_p / n
    media_a = media_a / n

    for i in zip(pred,act):
        sum1 += (i[0] - media_p)*(i[1] - media_a)
        sum2 += (i[0] - media_p)**2
        sum3 += (i[1] - media_a)**2

    sum2 = math.sqrt(sum2)
    sum3 = math.sqrt(sum3)
    pcc = sum1 / (sum2*sum3)
    return pcc

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
    
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Precc = TP / (TP + TN)
    Recall = TP / (TP + FN)
    
    return Acc, Precc, Recall

def advogato(G,fuente, sumidero):
    distancias = distanciaNodos(G)[fuente]
    distancias.pop(fuente)
    distancias.pop(sumidero)
    outdegree = dict(G.out_degree())
    avg_outdegree = {d: None for d in set(distancias.values())}
    advogato = {c: None for c in set(distancias.values())}
    advogato[0] = 1

    for d in avg_outdegree.keys():
        nodos = list(dict(filter(lambda x: x[1] == d, distancias.items())).keys())

        suma = 0
        x = len(nodos)

        for n in nodos:
            suma = suma + outdegree[n]

        avg_outdegree[d] = suma/x
        advogato[d] = max((advogato[d-1]/avg_outdegree[d]), .50)
            
    return advogato
    