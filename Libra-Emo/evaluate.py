import json
import os
from sklearn.metrics import classification_report

NEGS = ["ironic", "frustrated", "disgusted", "sad", "angry", "fearful", "hateful", "despairful"]
ALL = ["excited", "happy", "amazed", "surprised", "neutral", "ironic", "frustrated", "disgusted", "sad", "angry", "fearful", "hateful", "despairful"]

def compute_filtered_accuracy(metrics: dict, selected_classes: list) -> float:
    """
    Calculate accuracy for specified classes.
    Uses recall to compute accuracy for each class.
    """
    correct = 0
    total = 0
    for class_name in selected_classes:
        if class_name in metrics:
            recall = metrics[class_name]["recall"]  # Use recall to compute accuracy
            support = metrics[class_name]["support"]
            correct += recall * support
            total += support
    return correct / total if total > 0 else 0.0

def compute_filtered_macro_f1(metrics: dict, selected_classes: list) -> float:
    """
    Calculate macro F1 score for specified classes.
    """
    f1s = []
    for class_name in selected_classes:
        if class_name in metrics:
            f1s.append(metrics[class_name]["f1-score"])
    return sum(f1s) / len(f1s) if f1s else 0.0

def compute_filtered_weighted_f1(metrics: dict, selected_classes: list) -> float:
    """
    Calculate weighted F1 score for specified classes.
    """
    total_support = 0
    weighted_sum = 0
    for class_name in selected_classes:
        if class_name in metrics:
            f1 = metrics[class_name]["f1-score"]
            support = metrics[class_name]["support"]
            weighted_sum += f1 * support
            total_support += support
    return weighted_sum / total_support if total_support > 0 else 0.0

def compute_metrics(labels: list, preds: list, NEGS: list):
    """
    Compute and print various evaluation metrics.
    """
    report = classification_report(labels, preds, digits=4, output_dict=False)
    print(report)
    report = classification_report(labels, preds, digits=4, output_dict=True)   
    overall_acc = report['accuracy'] * 100
    overall_macro_f1 = report['macro avg']['f1-score'] * 100
    overall_weighted_f1 = report['weighted avg']['f1-score'] * 100
    neg_acc = compute_filtered_accuracy(report, NEGS) * 100
    neg_macro_f1 = compute_filtered_macro_f1(report, NEGS) * 100
    neg_weighted_f1 = compute_filtered_weighted_f1(report, NEGS) * 100
    print(f"overall_acc overall_macro_f1 overall_weighted_f1 neg_acc neg_macro_f1 neg_weighted_f1")
    print(f"{overall_acc:.2f} {overall_macro_f1:.2f} {overall_weighted_f1:.2f} {neg_acc:.2f} {neg_macro_f1:.2f} {neg_weighted_f1:.2f}")

def get_predictions(predict_root):
    """
    Load and deduplicate predictions from all JSONL files in the prediction directory.
    """
    predictions = []
    filter_set = set()
    for file in os.listdir(predict_root):
        if file.endswith(".jsonl"):
            with open(os.path.join(predict_root, file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    if data["video"] in filter_set:
                        continue
                    predictions.append(data)
                    filter_set.add(data["video"])
    return predictions

def get_label(answer):
    """
    Extract and validate the predicted label from the model's answer.
    Returns 'neutral' if the predicted label is not in the allowed set.
    """
    answer = answer.split("\n")[0].strip().lower()
    if answer not in ALL:
        print(f"Invalid prediction: {answer}")
        return "neutral"  # Return neutral if prediction is not in allowed set
    else:
        return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_root", type=str, required=True, help="Directory containing prediction JSONL files")
    args = parser.parse_args()

    predictions = get_predictions(args.predict_root)
    print(f"Number of predictions: {len(predictions)}")  # Should be 635 for test set

    labels = []
    preds = []
    for i, p in enumerate(predictions):
        labels.append(p["gt"])
        pred = get_label(p["answer"])
        preds.append(pred)

    compute_metrics(labels, preds, NEGS)