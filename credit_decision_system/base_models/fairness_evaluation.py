import matplotlib.pyplot as plt
import numpy as np

def visualize_fairness_metrics(model, X_test, y_test, protected_attribute, preprocessor=None):
    """
    Calculates and visualizes fairness metrics across protected attribute groups
    """
    # If preprocessor is provided, we need to get raw data
    test_data = X_test.copy()
    test_data['actual'] = y_test.values
    
    # Get predictions
    if preprocessor:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    test_data['predicted'] = y_pred
    test_data['probability'] = y_prob
    
    # Calculate metrics by group
    metrics = {}
    groups = test_data[protected_attribute].unique()
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # True positive rates by group
    tpr_by_group = []
    fpr_by_group = []
    acceptance_by_group = []
    
    # Calculate metrics for each group
    for group in groups:
        group_data = test_data[test_data[protected_attribute] == group]
        
        # True positive rate (Equal opportunity)
        tpr = (group_data[(group_data['actual'] == 0) & (group_data['predicted'] == 0)].shape[0] / 
              max(1, group_data[group_data['actual'] == 0].shape[0]))
        tpr_by_group.append((group, tpr))
        
        # False positive rate
        fpr = (group_data[(group_data['actual'] == 1) & (group_data['predicted'] == 0)].shape[0] / 
              max(1, group_data[group_data['actual'] == 1].shape[0]))
        fpr_by_group.append((group, fpr))
        
        # Acceptance rate (Demographic parity)
        acceptance_rate = group_data[group_data['predicted'] == 0].shape[0] / group_data.shape[0]
        acceptance_by_group.append((group, acceptance_rate))
    
    # Plot metrics
    metrics_to_plot = [
        (tpr_by_group, "True Positive Rate (Equal Opportunity)", ax[0]),
        (fpr_by_group, "False Positive Rate", ax[1]),
        (acceptance_by_group, "Acceptance Rate (Demographic Parity)", ax[2])
    ]
    
    for metric_data, title, axis in metrics_to_plot:
        groups, values = zip(*sorted(metric_data))
        axis.bar(groups, values)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        
        # Add horizontal line for reference
        axis.axhline(np.mean(values), color='red', linestyle='--')
        
        # Add value labels
        for i, v in enumerate(values):
            axis.text(i, v + 0.03, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.suptitle(f"Fairness Metrics by {protected_attribute}")
    plt.subplots_adjust(top=0.9)
    
    return plt, metrics_to_plot