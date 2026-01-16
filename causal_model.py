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