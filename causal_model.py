from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

from dataset import load_raw_dataset

# Define the Causal Graph (DAG) manual
# Arrows (->) represent "causes"
causal_graph = """
digraph {
    # Exogenous / Demographic Factors (Root causes)
    age -> education_num;
    age -> marital_status;
    age -> occupation;
    age -> hours_per_week;
    age -> income;

    sex -> education_num;
    sex -> occupation;
    sex -> hours_per_week;
    sex -> income;

    race -> education_num;
    race -> occupation;
    race -> income;

    native_country -> education_num;
    native_country -> income;

    # Intermediate / Societal Factors
    education_num -> occupation;
    education_num -> income;

    marital_status -> relationship;
    marital_status -> hours_per_week;
    marital_status -> income;

    relationship -> income;

    # Economic / Professional Factors
    workclass -> occupation;
    workclass -> income;

    occupation -> hours_per_week;
    occupation -> income;

    hours_per_week -> income;

    capital_gain -> income;
    capital_loss -> income;
}
"""


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