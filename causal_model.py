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

    # Visualization: Export the discovered graph to a PNG file
    print("Discovery complete. Exporting to 'discovered_dag.png'...")
    try:
        pyd = GraphUtils.to_pydot(cg.G)
        pyd.write_png('discovered_dag.png')
    except Exception as e:
        print(f"Visualization failed: {e}. Check if Graphviz is installed on your system.")

    return cg


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
