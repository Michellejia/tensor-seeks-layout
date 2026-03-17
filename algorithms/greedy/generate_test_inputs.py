
"""
Test Input Generator for Greedy Layout Selection Algorithm
Generates various test cases with different graph structures and complexities
"""

import json
from typing import List, Dict, Tuple
import random


class TestInputGenerator:
    """Generate test inputs for the layout selection algorithm"""
    
    @staticmethod
    def generate_linear_chain(num_nodes: int = 5, num_layouts: int = 3) -> Dict:
        """Generate a linear chain graph (node0 -> node1 -> node2 -> ...)"""
        nodes = []
        for i in range(num_nodes):
            nodes.append({
                "id": f"node_{i}",
                "operations": [f"matmul", f"add"],
                "type": "compute"
            })
        
        edges = []
        for i in range(num_nodes - 1):
            edges.append({
                "source": f"node_{i}",
                "target": f"node_{i+1}",
                "tensor_shape": [128, 128]
            })
        
        layout_options = {}
        for i in range(num_nodes):
            layouts = []
            for j in range(num_layouts):
                if j == 0:
                    axes = []
                elif j == 1:
                    axes = [0]
                else:
                    axes = [0, 1]
                
                layouts.append({
                    "id": j,
                    "partition_axes": axes,
                    "description": f"Layout_{j}"
                })
            layout_options[f"node_{i}"] = layouts
        
        return {
            "name": "linear_chain",
            "description": f"Linear chain with {num_nodes} nodes",
            "nodes": nodes,
            "edges": edges,
            "layout_options": layout_options
        }
    
    @staticmethod
    def generate_diamond_graph(num_layouts: int = 3) -> Dict:
        """Generate a diamond-shaped graph (fork-join pattern)"""
        nodes = [
            {"id": "input", "operations": ["load"], "type": "input"},
            {"id": "branch_a", "operations": ["matmul", "relu"], "type": "compute"},
            {"id": "branch_b", "operations": ["conv2d", "relu"], "type": "compute"},
            {"id": "merge", "operations": ["add", "softmax"], "type": "compute"},
            {"id": "output", "operations": ["store"], "type": "output"}
        ]
        
        edges = [
            {"source": "input", "target": "branch_a", "tensor_shape": [64, 64]},
            {"source": "input", "target": "branch_b", "tensor_shape": [64, 64]},
            {"source": "branch_a", "target": "merge", "tensor_shape": [64, 64]},
            {"source": "branch_b", "target": "merge", "tensor_shape": [64, 64]},
            {"source": "merge", "target": "output", "tensor_shape": [64, 64]}
        ]
        
        layout_options = {}
        for node in nodes:
            layouts = []
            for j in range(num_layouts):
                if j == 0:
                    axes = []
                elif j == 1:
                    axes = [0]
                else:
                    axes = [0, 1]
                
                layouts.append({
                    "id": j,
                    "partition_axes": axes,
                    "description": f"Layout_{j}"
                })
            layout_options[node["id"]] = layouts
        
        return {
            "name": "diamond_graph",
            "description": "Fork-join pattern with 2 parallel branches",
            "nodes": nodes,
            "edges": edges,
            "layout_options": layout_options
        }
    
    @staticmethod
    def generate_transformer_block(num_layouts: int = 4) -> Dict:
        """Generate a simplified transformer block structure"""
        nodes = [
            {"id": "input_embedding", "operations": ["embedding_lookup"], "type": "input"},
            {"id": "query_proj", "operations": ["matmul"], "type": "compute"},
            {"id": "key_proj", "operations": ["matmul"], "type": "compute"},
            {"id": "value_proj", "operations": ["matmul"], "type": "compute"},
            {"id": "attention_scores", "operations": ["matmul", "softmax"], "type": "compute"},
            {"id": "attention_output", "operations": ["matmul"], "type": "compute"},
            {"id": "ffn_1", "operations": ["matmul", "gelu"], "type": "compute"},
            {"id": "ffn_2", "operations": ["matmul"], "type": "compute"},
            {"id": "layer_norm", "operations": ["layer_norm"], "type": "compute"},
            {"id": "output", "operations": ["store"], "type": "output"}
        ]
        
        edges = [
            {"source": "input_embedding", "target": "query_proj", "tensor_shape": [32, 512]},
            {"source": "input_embedding", "target": "key_proj", "tensor_shape": [32, 512]},
            {"source": "input_embedding", "target": "value_proj", "tensor_shape": [32, 512]},
            {"source": "query_proj", "target": "attention_scores", "tensor_shape": [32, 512]},
            {"source": "key_proj", "target": "attention_scores", "tensor_shape": [32, 512]},
            {"source": "attention_scores", "target": "attention_output", "tensor_shape": [32, 512]},
            {"source": "value_proj", "target": "attention_output", "tensor_shape": [32, 512]},
            {"source": "attention_output", "target": "ffn_1", "tensor_shape": [32, 512]},
            {"source": "ffn_1", "target": "ffn_2", "tensor_shape": [32, 2048]},
            {"source": "ffn_2", "target": "layer_norm", "tensor_shape": [32, 512]},
            {"source": "layer_norm", "target": "output", "tensor_shape": [32, 512]}
        ]
        
        layout_options = {}
        for node in nodes:
            layouts = []
            for j in range(num_layouts):
                if j == 0:
                    axes = []  # No partitioning
                elif j == 1:
                    axes = [0]  # Batch dimension
                elif j == 2:
                    axes = [1]  # Sequence dimension
                else:
                    axes = [0, 1]  # Both dimensions
                
                layouts.append({
                    "id": j,
                    "partition_axes": axes,
                    "description": f"Layout_{j}"
                })
            layout_options[node["id"]] = layouts
        
        return {
            "name": "transformer_block",
            "description": "Simplified transformer attention + FFN block",
            "nodes": nodes,
            "edges": edges,
            "layout_options": layout_options
        }
    
    @staticmethod
    def generate_complex_dag(num_nodes: int = 10, edge_probability: float = 0.3, 
                            num_layouts: int = 3) -> Dict:
        """Generate a random complex DAG"""
        nodes = []
        for i in range(num_nodes):
            op_types = ["matmul", "conv2d", "add", "relu", "softmax", "layer_norm"]
            ops = random.sample(op_types, k=min(2, len(op_types)))
            nodes.append({
                "id": f"node_{i}",
                "operations": ops,
                "type": "compute"
            })
        
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_probability:
                    edges.append({
                        "source": f"node_{i}",
                        "target": f"node_{j}",
                        "tensor_shape": [random.choice([32, 64, 128]), 
                                       random.choice([32, 64, 128])]
                    })
        
        layout_options = {}
        for i in range(num_nodes):
            layouts = []
            for j in range(num_layouts):
                num_axes = random.randint(0, 2)
                axes = list(range(num_axes))
                layouts.append({
                    "id": j,
                    "partition_axes": axes,
                    "description": f"Layout_{j}"
                })
            layout_options[f"node_{i}"] = layouts
        
        return {
            "name": "complex_dag",
            "description": f"Random DAG with {num_nodes} nodes and ~{len(edges)} edges",
            "nodes": nodes,
            "edges": edges,
            "layout_options": layout_options
        }


def generate_all_test_inputs():
    """Generate all test input files"""
    generator = TestInputGenerator()
    
    test_cases = [
        ("test_linear_chain_small.json", generator.generate_linear_chain(5, 3)),
        ("test_linear_chain_large.json", generator.generate_linear_chain(10, 4)),
        ("test_diamond_graph.json", generator.generate_diamond_graph(3)),
        ("test_transformer_block.json", generator.generate_transformer_block(4)),
        ("test_complex_dag_small.json", generator.generate_complex_dag(8, 0.3, 3)),
        ("test_complex_dag_large.json", generator.generate_complex_dag(15, 0.25, 4)),
    ]
    
    print("Generating test input files...")
    print("=" * 60)
    
    for filename, test_data in test_cases:
        with open(filename, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✓ {filename}")
        print(f"  - Nodes: {len(test_data['nodes'])}")
        print(f"  - Edges: {len(test_data['edges'])}")
        print(f"  - Description: {test_data['description']}")
        print()
    
    print("=" * 60)
    print(f"Generated {len(test_cases)} test input files")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    generate_all_test_inputs()

