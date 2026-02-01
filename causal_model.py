import pydot
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

from dataset import load_raw_dataset

manual_causal_rules = [
    (["age"], ["marital-status"]),
    (["age"], ["workclass"]),
    (["age"], ["occupation"]),
    (["age"], ["hours-per-week"]),
    (["native-country"], ["workclass"]),

    (["education-num"], ["occupation"]),
    (["workclass"], ["occupation"]),
    (["occupation"], ["hours-per-week"]),

    (["marital-status"], ["relationship"]),

    (["education-num"], ["capital-gain"]),
    (["education-num"], ["capital-loss"]),
    (["occupation"], ["capital-gain"]),
    (["occupation"], ["capital-loss"]),
    (["hours-per-week"], ["capital-gain"]),
    (["hours-per-week"], ["capital-loss"]),
]


def get_learned_causal_rules(cg):
    """
    Translates the CausalGraph object from causal-learn into a list of rules:
    [( [parent], [child] ), ...]
    """
    rules = []
    nodes = cg.G.nodes
    # Get the adjacency matrix: 1 = directed edge, -1 = undirected
    adj = cg.G.graph

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            # PC: directed edge i -> j corresponds to adj[i,j] == -1 and adj[j,i] == 1
            if adj[i, j] == -1 and adj[j, i] == 1:
                parent_name = nodes[i].get_name()
                child_name = nodes[j].get_name()
                rules.append(([parent_name], [child_name]))

    return rules


def generate_causal_graph():
    """
    Analyzes the causal structure of the dataset using the PC (Peter-Clark) algorithm.
    """
    X, y = load_raw_dataset()

    # Combine features and target for holistic analysis
    df_discovery = X.copy()
    df_discovery['income'] = y.iloc[:, 0]

    # Cleanup: Make column names Graphviz-safe (replace hyphens with underscores)
    df_discovery.columns = [c.replace('-', '_') for c in df_discovery.columns]

    # Drop redundant or non-causal features
    to_drop = ['fnlwgt', 'education']
    df_discovery = df_discovery.drop(columns=[c for c in to_drop if c in df_discovery.columns])

    # Convert Categorical data to Numeric (Label Encoding)
    # The PC algorithm requires numerical input to perform independence tests.
    for col in df_discovery.select_dtypes(include=['object']).columns:
        df_discovery[col] = df_discovery[col].astype('category').cat.codes

    # Extract data and labels for causal-learn
    data = df_discovery.to_numpy()
    labels = df_discovery.columns.tolist()

    print("Starting Causal Discovery (PC Algorithm)...")

    # alpha=0.05 is the significance level for conditional independence tests.
    # The algorithm removes edges where variables are found to be independent.
    cg = pc(data, alpha=0.05, node_names=labels)
    return cg


def visualize_graph(rules, title, output_filename='dag.png'):
    """
    Creates a PNG visualization of the manually defined causal rules.
    """
    title = ""

    # Initialize a directed graph
    graph = pydot.Dot(graph_type='digraph', label=title, labelloc='t')

    graph.set('dpi', str(300))

    font_size = "26"

    graph.set_node_defaults(
        style='filled',
        fillcolor='white',
        fontsize=font_size,

        margin="0.2,0.1"
    )

    # Use a set to keep track of created nodes to avoid duplicates
    created_nodes = set()

    print(f"Generating manual graph visualization: {output_filename}...")

    for parents, children in rules:
        for p in parents:
            # Clean names for Graphviz (replace hyphens with underscores)
            p_name = p.replace('-', '_')
            if p_name not in created_nodes:
                graph.add_node(pydot.Node(p_name))
                created_nodes.add(p_name)

            for c in children:
                c_name = c.replace('-', '_')
                if c_name not in created_nodes:
                    graph.add_node(pydot.Node(c_name))
                    created_nodes.add(c_name)

                # Add the directed edge
                graph.add_edge(pydot.Edge(p_name, c_name))

    try:
        graph.write_png(output_filename)
        print(f"Manual graph successfully exported to '{output_filename}'.")
    except Exception as e:
        print(f"Visualization failed")


def get_given_dag_spec(feature_cols):
    """Return the hand-defined DAG as a dag_spec: [(child, [parents...]), ...]."""

    cleaned = []
    for parents, children in manual_causal_rules:
        for child in children:
            if child not in feature_cols:
                continue
            ps = [p for p in parents if p in feature_cols]
            if ps:
                cleaned.append((child, ps))
    return cleaned


if __name__ == "__main__":
    cg = generate_causal_graph()
    generated_rules = get_learned_causal_rules(cg)
    visualize_graph(rules=generated_rules, title="Generated causal graph", output_filename='generated_dag.png')
    visualize_graph(rules=manual_causal_rules, title="Manual defined causal graph", output_filename="manual_dag.png")
