df=pd.read_csv("ua_sentiment_dataset_labeled_lang.csv")
f["sentiment_deepseek"] = sentiment_cleaned
df["annotator_response"] = df["annotator_response"].str.strip().str.lower()
df["sentiment_deepseek"] = df["sentiment_deepseek"].str.strip().str.lower()
df_filtered = df[df["annotator_response"].str.lower() != "idk"].copy()

from sklearn.metrics import precision_recall_fscore_support, classification_report
import pandas as pd

def evaluate_sentiment(df, y_true_col, y_pred_col, group_col="language"):
    """
    Evaluate sentiment classification with overall and per-language-group metrics.
    
    Params:
    - df: pd.DataFrame containing predictions and true labels
    - y_true_col: column name of true labels (e.g. human annotations)
    - y_pred_col: column name of model predictions (e.g. DeepSeek output)
    - group_col: column to group by (e.g. 'language')

    Returns:
    - dict with overall metrics and per-group metrics
    """
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]

    # Overall metrics
    overall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    overall_micro = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    result = {
        "overall": {
            "macro": {
                "precision": overall_macro[0],
                "recall": overall_macro[1],
                "f1": overall_macro[2],
            },
            "micro": {
                "precision": overall_micro[0],
                "recall": overall_micro[1],
                "f1": overall_micro[2],
            }
        },
        "by_group": {}
    }

    # Per-language group metrics
    for group_value in df[group_col].unique():
        subset = df[df[group_col] == group_value]
        if subset.empty:
            continue

        group_true = subset[y_true_col]
        group_pred = subset[y_pred_col]

        macro = precision_recall_fscore_support(group_true, group_pred, average='macro', zero_division=0)
        micro = precision_recall_fscore_support(group_true, group_pred, average='micro', zero_division=0)

        result["by_group"][group_value] = {
            "macro": {
                "precision": macro[0],
                "recall": macro[1],
                "f1": macro[2],
            },
            "micro": {
                "precision": micro[0],
                "recall": micro[1],
                "f1": micro[2],
            }
        }

    return result

metrics = evaluate_sentiment(df_filtered, y_true_col="annotator_response", y_pred_col="sentiment_deepseek")

import pprint
pprint.pprint(metrics)
