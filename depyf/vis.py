from depyf.depyf import decompose_basic_blocks

def visualize_cfg(code: CodeType):
    """Visualize the control flow graph of a code object."""
    blocks = decompose_basic_blocks(code)
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    for block in blocks:
        G.add_node(block.code_range())
    for block in blocks:
        for to_block in block.to_blocks:
            G.add_edge(block.code_range(), to_block.code_range())
        for from_block in block.from_blocks:
            G.add_edge(from_block.code_range(), block.code_range())
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1000)
    nx.draw_networkx_edges(G, pos, node_size=1000)
    nx.draw_networkx_labels(G, pos)
    plt.show()
