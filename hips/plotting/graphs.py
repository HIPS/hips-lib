# From https://groups.google.com/forum/#!topic/networkx-discuss/FwYk0ixLDuY

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

def draw_curvy_network(G, pos, ax, 
                       node_radius=0.02, node_color='b', node_edge_color='b', node_alpha=0.5, 
                       edge_color='k', edge_alpha=0.5, edge_width=2):
    assert isinstance(G, nx.Graph), "G must be a NetworkX graph!"
    
    # Convert node colors to lists
    def _to_list(x, N):
        if isinstance(x, list):
            assert len(x) == N
            return x
        else:
            return [x] * N

    node_radius = _to_list(node_radius, len(G.nodes()))
    node_color = _to_list(node_color, len(G.nodes()))
    node_edge_color = _to_list(node_edge_color, len(G.nodes()))
    node_alpha = _to_list(node_alpha, len(G.nodes()))

    edge_color = _to_list(edge_color, len(G.edges()))
    edge_alpha = _to_list(edge_alpha, len(G.edges()))
    edge_width = _to_list(edge_width, len(G.edges()))
    
    # Plot the nodes
    for n,r,a,fc,ec in zip(G, node_radius, node_alpha, node_color, node_edge_color):
        c=Circle(pos[n], radius=r, alpha=a, fc=fc, ec=ec)
        ax.add_patch(c)
        G.node[n]['patch']=c


    # Plot the edges
    seen={}
    for (u,v,d),a,lw,ec in zip(G.edges(data=True), edge_alpha, edge_width, edge_color):
        n1=G.node[u]['patch']
        n2=G.node[v]['patch']
        rad=-0.1
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1

        e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=10.0,
                            lw=lw,
                            alpha=a,
                            color=ec)
        seen[(u,v)]=rad
        ax.add_patch(e)

    return e


if __name__ == "__main__":
    G=nx.MultiDiGraph([(1,2),(1,2),(2,3),(3,4),(2,4),
                       (1,2),(1,2),(1,2),(2,3),(3,4),(2,4)]
                      )

    pos=nx.spring_layout(G)
    ax=plt.gca()
    draw_curvy_network(G,pos,ax,node_color='r', node_edge_color='r')
    ax.autoscale()
    plt.axis('equal')
    plt.axis('off')
    #plt.savefig("graph.pdf")
    plt.show()
