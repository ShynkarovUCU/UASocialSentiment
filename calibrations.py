import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
model_name='complete'
#model_name='mixed'
#model_name='complete'


def plot_grouped_calibration_curves_with_ece(datasets, dataset_names):
    # Define bins and calculate the bin centers
    
    ece_llama=calculate_ece(all_val_labels1, all_val_preds1, all_val_probs1, n_bins=10)
    ece_mixed=calculate_ece(all_val_labels2, all_val_preds2, all_val_probs2, n_bins=10)
    ece_gptwiki=calculate_ece(all_val_labels3, all_val_preds3, all_val_probs3, n_bins=10)
    ece_mistral=calculate_ece(all_val_labels4, all_val_preds4, all_val_probs4, n_bins=10)
    eces=[ece_llama,ece_mixed,ece_gptwiki,ece_mistral]
    ece_all=calculate_ece(all_val_labels, all_val_preds, all_val_probs, n_bins=10)

    bins = np.linspace(0.5, 1.0, num=6)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = (bins[1] - bins[0]) / (len(datasets) + 1)

    fig = go.Figure()
    overall_ece = 0
    total_samples_all_datasets = 0

    for i, ((true_labels, pred_labels, probabilities), name) in enumerate(zip(datasets, dataset_names)):
        offset = ((i - len(datasets) / 2) * bar_width) + (bar_width / 2)
        confidences = [prob[pred] for pred, prob in zip(pred_labels, probabilities)]
        bin_indices = np.digitize(confidences, bins, right=True)

        cumulative_accuracies = []
        bin_counts = []
        bin_weights = []
        for bin_index in range(1, len(bins)):
            indices = [j for j, b in enumerate(bin_indices) if b == bin_index]
            bin_counts.append(len(indices))
            if indices:
                correct_predictions = [1 if true_labels[j] == pred_labels[j] else 0 for j in indices]
                cumulative_accuracy = np.mean(correct_predictions)
                cumulative_accuracies.append(cumulative_accuracy * 100)
                bin_weights.append(len(indices) / len(confidences))
            else:
                cumulative_accuracies.append(0)
                bin_weights.append(0)

        # Compute ECE for each dataset
        ece = sum([bin_weights[j] * abs(cumulative_accuracies[j]/100 - ((bins[j] + bins[j-1]) / 2)) 
                   for j in range(len(bin_weights))])
        overall_ece += ece * len(confidences)
        total_samples_all_datasets += len(confidences)

        # Add the ECE as a data label
        fig.add_trace(go.Bar(
            x=[bin_centers[i] + offset for i in range(len(bin_centers))],
            y=cumulative_accuracies,
            width=bar_width,
            name=f'{name} (ECE: {eces[i]:.2f})',
            opacity=0.6
        ))

    # Calculate overall ECE for all datasets
    bin_accuracy = [50, 60, 70, 80,90]  # Replace with your actual accuracy data for each bin

    # Assuming bins contains the bin edges
    # For example: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bins = np.linspace(0.5, 1.0, num=6)  # This should match your actual bins

    staircase_x = []
    staircase_y = []

    for i in range(len(bins) - 1):
        # Starting point of the horizontal line (left edge of the bin)
        staircase_x.append(bins[i])
        staircase_y.append(bin_accuracy[i])

        # Ending point of the horizontal line (right edge of the bin)
        staircase_x.append(bins[i + 1])
        staircase_y.append(bin_accuracy[i])

    # Add the final point for the last bin
    staircase_x.append(bins[-1])
    staircase_y.append(bin_accuracy[-1])

    # Add the staircase calibration line to the plot
    fig.add_trace(go.Scatter(
        x=staircase_x,
        y=staircase_y,
        mode='lines',
        line=dict(color='black', width=5, dash='dash'),
        name='Calibration Line'
    ))

    # Modify the legend to show only one "Staircase Calibration Line" entry
    # This can be done by adding `showlegend=False` to all but one of the staircase traces.

    # Annotation for the Staircase Calibration Line

# Remove the legend for all the other lines if not needed




    # Formatting the plot
    fig.update_layout(
        plot_bgcolor='white',  # Sets the plot background to white
        paper_bgcolor='white',  # Sets the surrounding paper area to white
        legend_title_font_size=40*2,
        legend_font_size=35*2,
        xaxis=dict(title="Confidence Interval"),
        xaxis_title_font_size=40*2,
        yaxis=dict(title="Accuracy (%)", range=[0, 100]),
        yaxis_title_font_size=40*2,
        barmode='group',
        font=dict(
        family="Times Roman",
        size=35*2,
        color="black"
    )
    )


    #fig.write_image("calibration_" + model_name + ".pdf", width=1920, height=1080, scale=2)
    fig.write_image("NEW_calibration_" + model_name + ".png", width=1920, height=1080)

    fig.show()

plot_grouped_calibration_curves_with_ece(datasets, dataset_names)