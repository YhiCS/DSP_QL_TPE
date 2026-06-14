import pandas as pd;
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Resolved relative to this file so the module works from any CWD.
_HERE    = Path(__file__).resolve().parent
MAPS_DIR = _HERE / "maps_and_settings"

def loadMap (map_type = 'OG', pair = None, derailPenalty = -1e-5, goalReward = 100, disReward = 50, is_tour = False):
    '''
    Load maps from csv and generate graph, start, and goal nodes
    Avaliable Maps: 'OG' (default), 'Gal_Cue', 'Gal_Cat'

    "OG" Pairs: dsp1_02, dsp1_03 ... dsp1_tour, dsp2_02, dsp2_03... dsp2_tour

    "Gal_Cue" Pairs: A2D, A3D, R2D, R3D, tour
    
    "Gal_Cat" Pairs: Adv, Dis
    '''

    norMap = pd.read_csv(MAPS_DIR / "dsp_nor_SL.csv", header=None).values.tolist()
    altMap = pd.read_csv(MAPS_DIR / "dsp_alt_SL.csv", header=None).values.tolist()
    galMap = pd.read_csv(MAPS_DIR / "dsp_gal_SL.csv", header=None).values.tolist()

    DSPTrialPair = pd.read_csv(MAPS_DIR / 'DSPTrialPairs.csv')
    DSPGalPair = pd.read_csv(MAPS_DIR / 'DSPGalPairs.csv')
    conList = pd.read_csv(MAPS_DIR / 'connections_SLnodes.csv').set_index(['map', 'tarNode'])

    currentMap = disObj = startnode = []
    mapKey = goalObj = startObj = startLoc = 0

    if map_type == "OG":
        ##### OG DSP map
        if not pair:
            currentPair = 'dsp2_24'
        elif pair not in list(DSPTrialPair.loc[:,'Trial'])+['dsp1_tour', 'dsp2_tour']:
            raise Exception(f"Wrong pair! This is the OG DSP. Current Pair: '{pair}' is not for the current map!")
        else:
            currentPair = pair        
        currentMap = norMap if "dsp1" in currentPair else altMap
        mapKey = 'norMap' if "dsp1" in currentPair else 'altMap'

        if not is_tour:
            startObj = DSPTrialPair.loc[DSPTrialPair['Trial'] ==currentPair, 'Start'].values[0]
            startLoc = conList.loc[mapKey,startObj].values[0] # type: ignore
        else: 
            startObj = 0

        goalObj = [DSPTrialPair.loc[DSPTrialPair['Trial'] ==currentPair, 'Goal'].values[0]]

        # if not "tour" in pair: # type: ignore
        #     startObj = DSPTrialPair.loc[DSPTrialPair['Trial'] ==currentPair, 'Start'].values[0]
        #     startLoc = conList.loc[mapKey,startObj].values[0] # type: ignore
        #     goalObj = [DSPTrialPair.loc[DSPTrialPair['Trial'] ==currentPair, 'Goal'].values[0]]
        # ### Tour route
        # elif pair == "dsp1_tour":
        #     startObj = 0
        #     goalObj = DSPTrialPair.loc[DSPTrialPair['Trial'].str.contains('dsp1'), 'Goal'].values.tolist()
        # elif pair == "dsp2_tour":
        #     startObj = 0
        #     goalObj = DSPTrialPair.loc[DSPTrialPair['Trial'].str.contains('dsp2'), 'Goal'].values.tolist()        

    elif map_type == "Gal_Cue":
    ##### DSP Gallery Cued retrieval map
        if not pair:
            currentPair = DSPGalPair['Trial'] == 'A2D'
        elif pair not in ['A2D', 'A3D', 'R2D', 'R3D','gal_tour']:
            raise Exception(f"Wrong pair! This is the DSP Gallery Cue retrival task, Current Pair: '{pair}' is not for the current map!")
        else:
            currentPair = DSPGalPair['Trial'] == pair    
        currentMap = galMap
        mapKey = 'galMap'

        if not is_tour:
            startObj = DSPGalPair.loc[currentPair, 'Start'].values[0]
        else: 
            startObj = 0
        
        goalObj = DSPGalPair.loc[currentPair, 'Goals'].values.tolist()
        
        # if not "tour" in pair: # type: ignore
        #     startObj = DSPGalPair.loc[currentPair, 'Start'].values[0]
        #     goalObj = DSPGalPair.loc[currentPair, 'Goals'].values.tolist()
        # elif pair == 'gal_tour':
        #     startObj = 0
        #     goalObj = DSPGalPair['Goals'].unique().tolist()


    elif map_type == "Gal_Cat":
    ##### DSP Gallery category retrieval map
        AdvPair = DSPGalPair['Trial'] == "Adv"
        DisPair = DSPGalPair['Trial'] == "Dis"
        currentMap = galMap
        mapKey = 'galMap'
        startObj = DSPGalPair.loc[AdvPair, 'Start'] .values[0] if not is_tour else 0
        disObj = DSPGalPair.loc[DisPair, 'Goals'].values.tolist()

        if not pair:
            currentPair = 'Adv' 
        elif pair not in ['Adv', 'Dis']:
            raise Exception(f"Wrong pair! This is DSP Gallery Category retrival task, Current Pair: '{pair}' is not for the current map!")
        else:
            currentPair = pair    

        if currentPair  == "Adv":
        ## Advantage deck
            goalObj = DSPGalPair.loc[AdvPair, 'Goals'].values.tolist()
        elif currentPair  == "Dis":
        ## Disadvantage deck
            goalObj = DSPGalPair.loc[DisPair, 'Goals'].values.tolist()  

        

    #### Map-graph setup        
    G = nx.Graph()
    rows = len(currentMap)
    cols = len(currentMap[0])
    goalnodes=[]
    disNodes=[]
    blockage = 0 # front:1 back: -1

    for i in range(rows):
        for j in range(cols):
            cell = currentMap[i][j]
            if cell != '#':  # Not a wall
                node = (i, j)
                if cell in goalObj:
                    goalnodes.append(node)
                    reward = goalReward
                elif ((len(disObj) != 0) & (cell in disObj)):
                    disNodes.append(node)
                    reward = disReward

                elif cell == ".": reward = derailPenalty
                elif cell == int(startLoc) + blockage: reward = -100
                else: reward = 0

                G.add_node(node, label = cell, reward=reward)
                
                # Possible moves: right, down, left, up
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + dx, j + dy
                    if (0 <= ni < rows and 0 <= nj < cols
                        and currentMap[ni][nj] != '#' 
                        and not currentMap[ni][nj].isalpha()):
                        if not cell.isalpha():
                            neighbor = (ni, nj)
                            G.add_edge(node, neighbor, cost=1)
                        elif str(currentMap[ni][nj]) == str(conList.loc[(mapKey, cell),'routeNode']):
                            neighbor = (ni, nj)
                            G.add_edge(node, neighbor, cost=1)
    routenodes = [node for node, attr in G.nodes(data=True) if attr.get('label', 0).isnumeric()]
    nodeLabs = nx.get_node_attributes(G, 'label')
    startnode = next((key for key in nodeLabs if nodeLabs[key] == str(startLoc)), None) if startObj != 'St' else (2,11)
    pos = {node: (node[1], -node[0]) for node in G.nodes()}

    return G, pos, goalnodes, startnode, disNodes, routenodes


def plot_route(pair = None, derailPenalty = -1, default_Q_Strength = 1e-5, route_nodes = None, RouteNodeIdx = None, replacing = False, subjID = None):
    map_type = "OG"
    G, pos, goalnodes, startnode, disNodes, routenodes = loadMap(map_type, pair, 
    derailPenalty=-1, goalReward=100, disReward=-50)    
    node_index = {i: n for i, n in enumerate(sorted(G.nodes, key=lambda n: str(n)))}

    if (route_nodes == None):
        if (RouteNodeIdx != None):
            route_nodes = [node_index[ri] for ri in RouteNodeIdx]
        else:
            raise ValueError("No route nodes were provided!")
    

    plt.figure(figsize=(5, 5))
    pos_plot = {node: (node[1], -node[0]) for node in G.nodes()}

    # nodes
    nx.draw_networkx_nodes(G, pos_plot, nodelist=goalnodes, node_color="red", node_size=100)
    nx.draw_networkx_nodes(G, pos_plot, nodelist=disNodes, node_color="cyan", node_size=100)
    nx.draw_networkx_nodes(G, pos_plot, nodelist=routenodes, node_color="yellow", node_size=20)
    nx.draw_networkx_nodes(G, pos_plot, nodelist=[startnode], node_color="lightgreen", node_size=100)

    # edges + labels
    nx.draw_networkx_edges(G, pos_plot)
    nx.draw_networkx_labels(G, pos_plot, nx.get_node_attributes(G, "label"), font_size=10)

    # arrows
    alphaSeq = np.linspace(0.3,1.0,len(route_nodes))
    # if any(i in route_nodes for i in pos) and replacing:
    #     route_nodes = replace_missing_with_nearest(route_nodes, G)
    derailSteps = 0
    for i in range(len(route_nodes)-1):
        derailSteps += 1 if not route_nodes[i] in routenodes else 0
        start = pos_plot[route_nodes[i]]
        end = pos_plot[route_nodes[i+1]]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(
            start[0], start[1],
            dx * 0.85, dy * 0.85,
            head_width=0.5, head_length=0.51,
            fc="green", ec="green",
            alpha = alphaSeq[i],
            length_includes_head=True
        )
    totalSteps = len(route_nodes)
    plt.title(f"Subj[{"Q" if subjID == None else subjID}] in '{pair}' Derail Steps:{derailSteps}; Total Steps:{totalSteps}")
    
    plt.axis("off")