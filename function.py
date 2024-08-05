from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def evaluate_ratings(preds_list, truths_list):
    mse_results = {"Total" : float, "Skill": float, "Experience": float, "Total_Experience": float, "Experience_with_Current_Company": float, "Education": float, "additional_considerations": float}

    total_pred = {"Justification": [],"Total":[], "Skill": [], "Experience": [], "Total_Experience":[], "Experience_with_Current_Company":[], "Education": [], "additional_considerations": []}
    total_actual = {"Justification": [],"Total":[], "Skill": [], "Experience": [], "Total_Experience":[], "Experience_with_Current_Company":[], "Education": [], "additional_considerations": []}

    for i in range(len(preds_list)):
        preds = preds_list[i]
        truths = truths_list[i]

        total_pred["Total"].append(preds["Total"])
        total_actual["Total"].append(truths["Total"])

        total_pred["Skill"].append(preds["Skill"])
        total_actual["Skill"].append(truths["Skill"])

        total_pred["Experience"].append(preds["Experience"])
        total_actual["Experience"].append(truths["Experience"])

        total_pred["Total_Experience"].append(preds["Total_Experience"])
        total_actual["Total_Experience"].append(truths["Total_Experience"])

        total_pred["Experience_with_Current_Company"].append(preds["Experience_with_Current_Company"])
        total_actual["Experience_with_Current_Company"].append(truths["Experience_with_Current_Company"])

        total_pred["Education"].append(preds["Education"])
        total_actual["Education"].append(truths["Education"])

        total_pred["additional_considerations"].append(preds["additional_considerations"])
        total_actual["additional_considerations"].append(truths["additional_considerations"])

        total_pred["Justification"].append(preds["Justification"])
        total_actual["Justification"].append(truths["Justification"])

    mse_results["Total"] = mean_squared_error(total_actual["Total"] , total_pred["Total"])
    mse_results["Skill"] = mean_squared_error(total_actual["Skill"] , total_pred["Skill"])
    mse_results["Experience"] = mean_squared_error(total_actual["Experience"] , total_pred["Experience"])
    mse_results["Total_Experience"] = mean_squared_error(total_actual["Total_Experience"] , total_pred["Total_Experience"])
    mse_results["Experience_with_Current_Company"] = mean_squared_error(total_actual["Experience_with_Current_Company"] , total_pred["Experience_with_Current_Company"])
    mse_results["Education"] = mean_squared_error(total_actual["Education"] , total_pred["Education"])
    mse_results["additional_considerations"] = mean_squared_error(total_actual["additional_considerations"] , total_pred["additional_considerations"])

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = [scorer.score(ref, hyp) for ref, hyp in zip(total_actual["Justification"], total_pred["Justification"])]

    # Average ROUGE scores (example for ROUGE-1)
    avg_scores = {
    'rouge1': sum(score['rouge1'].fmeasure for score in scores) / len(scores),
    'rouge2': sum(score['rouge2'].fmeasure for score in scores) / len(scores),
    'rougeL': sum(score['rougeL'].fmeasure for score in scores) / len(scores)
}

    return avg_scores, mse_results

actual = [
    """{
        "Skill": 9,
        "Experience": 10,
        "Total_Experience": 7,
        "Experience_with_Current_Company": 6,
        "Education": 9,
        "additional_considerations": 6,
        "Total": 8,
        "Justification": "The candidate shows a balanced experience across different areas and a solid educational foundation, with a slightly higher rating in additional considerations."
    }
    """]

pred = ["""
    {
        "Skill": 8,
        "Experience": 7,
        "Total_Experience": 6,
        "Experience_with_Current_Company": 5,
        "Education": 9,
        "additional_considerations": 4,
        "Total": 7,
        "Justification": "The candidate has strong skills and a good educational background but has relatively less experience with the current company."
    }
    """]

pred_json = []
truth_json = []

for i in range(len(pred)):
    pred_json.append(json.loads(pred[i]))
    truth_json.append(json.loads(actual[i]))

avg_rouge,mse_results = evaluate_ratings(pred_json, truth_json)

print(f'MSE_scores: {mse_results}')
print(f"Average ROUGE-1 score: {avg_rouge['rouge1']:.4f}")
print(f"Average ROUGE-2 score: {avg_rouge['rouge2']:.4f}")
print(f"Average ROUGE-L score: {avg_rouge['rougeL']:.4f}")


## ROUGE-1: Measures the overlap of unigrams (single words) between the generated and reference texts. It assesses how many individual words in the generated text appear in the reference text, evaluating overall word coverage.
## ROUGE-2: Measures the overlap of bigrams (two consecutive words) between the generated and reference texts. It assesses how many pairs of consecutive words in the generated text match those in the reference text, focusing on capturing word sequences and coherence.
## ROUGE-L: Measures the longest common subsequence between the generated and reference texts. It evaluates the longest sequence of words that appear in both texts in the same order, assessing fluency and coherence while accounting for the overall structure of the text.
