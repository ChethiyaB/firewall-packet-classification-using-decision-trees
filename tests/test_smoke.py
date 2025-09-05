from firewall_ml.model_zoo import make_model
def test_model_factory():
    assert make_model("decision_tree", {}).__class__.__name__ == "DecisionTreeClassifier"
