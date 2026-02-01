"""
ON-CHAIN GRAPH INTELLIGENCE ENGINE
Analyzes the topology of transaction flows to detect Smart Money clusters.
Uses NetworkX for graph structure and GUDHI for topological feature extraction.
"""

import networkx as nx
import numpy as np
import gudhi
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

@dataclass
class GraphSignature:
    num_clusters: int
    max_clique_size: int
    flow_persistence: float  # Persistence of money flow loops
    centrality_entropy: float # Diversity of important actors
    smart_money_score: float # 0-100 score

class WalletGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.smart_wallets = set()
        
    def update_graph(self, transactions: List[Dict]):
        """
        Update graph with new transactions.
        transactions: list of {from, to, value, timestamp}
        """
        for tx in transactions:
            self.graph.add_edge(tx['from'], tx['to'], weight=tx['value'], timestamp=tx['timestamp'])
            
        # Prune old edges to keep graph relevant (sliding window)
        # For simulation, we'll just keep the graph size manageable
        if self.graph.number_of_edges() > 1000:
            edges = list(self.graph.edges(data=True))
            edges.sort(key=lambda x: x[2]['timestamp'])
            self.graph.remove_edges_from([(e[0], e[1]) for e in edges[:200]])

    def compute_topology(self) -> GraphSignature:
        """
        Compute topological features of the wallet graph.
        """
        if self.graph.number_of_nodes() < 5:
            return GraphSignature(0, 0, 0.0, 0.0, 0.0)
            
        # 1. Cluster Analysis (Community Detection)
        # Using simple connected components for directed graph (weakly connected)
        clusters = list(nx.weakly_connected_components(self.graph))
        num_clusters = len(clusters)
        
        # 2. Clique Analysis (Dense subgroups)
        # Finding cliques in directed graph is hard, treat as undirected for this metric
        undirected = self.graph.to_undirected()
        # Max clique approximation
        try:
            max_clique = nx.graph_clique_number(undirected)
        except:
            max_clique = 0
            
        # 3. Flow Persistence (Topological Loops)
        # We convert the graph to a simplex tree where:
        # - Vertices are wallets (as INTEGERS)
        # - Edges are weighted by 1/value (high value = short distance)
        
        # Map wallet addresses to integers
        nodes = list(self.graph.nodes())
        node_to_int = {node: i for i, node in enumerate(nodes)}
        
        st = gudhi.SimplexTree()
        
        for u, v, data in self.graph.edges(data=True):
            # Invert value for filtration: bigger transfer = 'closer' connection
            weight = 1.0 / (data['weight'] + 1e-6)
            # Convert nodes to integers
            u_int = node_to_int[u]
            v_int = node_to_int[v]
            st.insert([u_int, v_int], filtration=weight)
            
        st.initialize_filtration()
        st.persistence()
        
        # H1 persistence (loops)
        h1 = st.persistence_intervals_in_dimension(1)
        flow_persistence = 0.0
        if len(h1) > 0:
            # Sum of lifetimes of loops
            flow_persistence = np.sum(h1[:, 1] - h1[:, 0])
            
        # 4. Centrality Entropy (Concentration of power)
        try:
            centrality = nx.pagerank(self.graph)
            values = np.array(list(centrality.values()))
            values = values / np.sum(values)
            centrality_entropy = -np.sum(values * np.log(values + 1e-10))
        except:
            centrality_entropy = 0.0
            
        # 5. Smart Money Score
        # Heuristic combination
        # High flow persistence + High max clique = Coordinated Smart Money
        smart_money_score = min(100, (flow_persistence * 10 + max_clique * 5))
        
        return GraphSignature(
            num_clusters=num_clusters,
            max_clique_size=max_clique,
            flow_persistence=flow_persistence,
            centrality_entropy=centrality_entropy,
            smart_money_score=smart_money_score
        )

    def simulate_data(self, num_tx=50):
        """
        Generate synthetic transaction data for demonstration.
        Creates a 'Smart Money' cluster pattern.
        """
        txs = []
        
        # Create a "Smart Money" ring
        smart_cluster = [f"0xSmart{i}" for i in range(5)]
        for i in range(len(smart_cluster)):
            txs.append({
                'from': smart_cluster[i],
                'to': smart_cluster[(i+1)%len(smart_cluster)],
                'value': random.uniform(50000, 200000), # High value
                'timestamp': 1234567890
            })
            
        # Random noise
        for _ in range(num_tx - 5):
            txs.append({
                'from': f"0xUser{random.randint(0, 20)}",
                'to': f"0xUser{random.randint(0, 20)}",
                'value': random.uniform(100, 5000), # Low value
                'timestamp': 1234567890
            })
            
        self.update_graph(txs)

    def update_and_get_signal(self, current_price: float) -> float:
        """
        Update graph with latest on-chain data and return Smart Money Score.
        """
        # In production, this would fetch real mempool/block data
        # For now, we simulate activity based on price action (mock)
        
        # Simulate more activity if price is high (FOMO)
        num_tx = int(50 + (current_price / 10000))
        self.simulate_data(num_tx=min(num_tx, 100))
        
        # Compute topology
        sig = self.compute_topology()
        
        return sig.smart_money_score

# Example Usage
if __name__ == "__main__":
    wg = WalletGraph()
    wg.simulate_data()
    sig = wg.compute_topology()
    print(f"Smart Money Score: {sig.smart_money_score:.2f}")
    print(f"Flow Persistence: {sig.flow_persistence:.4f}")
    print(f"Max Clique: {sig.max_clique_size}")
