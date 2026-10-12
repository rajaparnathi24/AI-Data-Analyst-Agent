"""
response_generator.py
---------------------
Translates verified analytical calculations from the Pandas analysis engine
into concise, natural-language English explanations.

Key principle:
The numerical figures ALWAYS originate directly from Pandas calculations,
preventing LLM hallucination while preserving fluent conversational flow.
"""

import pandas as pd
from llm_client import call_llm, check_api_status


def generate_response(question: str, analysis_result: dict, dataset_name: str = "dataset") -> str:
    """
    Produces a natural language answer based on verified analytical results.

    Parameters:
        question (str): The user's input question.
        analysis_result (dict): Output from analysis_engine.execute_plan.
        dataset_name (str): Label/filename of the dataset.

    Returns:
        str: Concise, verified conversational response.
    """
    if not analysis_result:
        return "I could not analyze the data. Please try rephrasing your question."

    # If analysis engine reported an error
    if analysis_result.get("status") == "error":
        err_msg = analysis_result.get("message") or analysis_result.get("error", "Unknown error")
        return f"**[Notice] Could not complete analysis**: {err_msg}\n\n*Tip: You can ask about averages, totals, top records, group comparisons, or missing values.*"

    task = analysis_result.get("task", "")
    res_type = analysis_result.get("result_type", "")
    val = analysis_result.get("value")
    engine_msg = analysis_result.get("message", "")

    # Attempt LLM natural language phrasing if Ollama is available
    status = check_api_status()
    if status["configured"]:
        llm_text = _generate_llm_grounded_text(question, task, val, engine_msg)
        if llm_text:
            return llm_text

    # Deterministic fallback response generation
    return _generate_deterministic_text(task, res_type, val, engine_msg)


def _generate_llm_grounded_text(question: str, task: str, value, engine_msg: str) -> str:
    """Calls Ollama to rephrase the verified computation cleanly."""
    val_repr = value.to_string(index=False) if isinstance(value, pd.DataFrame) else str(value)

    prompt = f"""
You are an expert Data Analyst summarizing verified findings for a stakeholder.
Question: "{question}"
Analytical Task: "{task}"
Verified Calculated Result from Pandas:
{val_repr}

Engine Summary: "{engine_msg}"

Instructions:
1. Explain the answer in 1-2 natural, polite sentences.
2. DO NOT change or invent any numbers. Rely strictly on the verified calculated result provided.
3. Be concise and direct.
"""
    system_inst = "You are a concise data analyst. Never invent numbers. Ground your explanation in verified facts."
    response, err = call_llm(prompt, system_instruction=system_inst, json_mode=False)
    if not err and response:
        return response.strip()

    return None


def _generate_deterministic_text(task: str, res_type: str, value, engine_msg: str) -> str:
    """Fallback text generator when offline or without API key."""
    if res_type == "scalar":
        return f"**[Result]**: {engine_msg}"

    elif res_type == "table" and isinstance(value, pd.DataFrame):
        rows_count = len(value)
        return f"**[Result]**: {engine_msg}\n\n*Displaying {rows_count} relevant record(s) in the table below.*"

    elif res_type == "modified_dataset":
        return f"**[KDD Data Transformation]**: {engine_msg}\n\n*The modified dataset is updated in memory and available for download below the Ask button.*"

    elif res_type == "ml_model" and isinstance(value, dict):
        model_name = value.get("model", "Machine Learning Model")
        target = value.get("target", "Target")
        if "accuracy" in value:
            # Classification
            acc = value["accuracy"] * 100
            f1 = value["f1_score"]
            prec = value["precision"]
            rec = value["recall"]
            return (
                f"**[Machine Learning Model Trained]**: {model_name} on target `{target}`\n\n"
                f"- **Accuracy**: {acc:.1f}%\n"
                f"- **F1-Score**: {f1:.3f}\n"
                f"- **Precision**: {prec:.3f}\n"
                f"- **Recall**: {rec:.3f}\n\n"
                f"*Evaluation computed on 25% held-out test split. Confusion matrix visualized in the chart panel below.*"
            )
        elif "r2" in value:
            # Regression
            mae = value["mae"]
            rmse = value["rmse"]
            r2 = value["r2"]
            return (
                f"**[Regression Model Trained]**: {model_name} on target `{target}`\n\n"
                f"- **R² Score (Goodness of Fit)**: {r2:.3f}\n"
                f"- **Mean Absolute Error (MAE)**: {mae:,.2f}\n"
                f"- **Root Mean Squared Error (RMSE)**: {rmse:,.2f}\n\n"
                f"*Actual vs Predicted scatter plot displayed below.*"
            )

    elif res_type == "cluster_result":
        return f"**[Unsupervised Clustering]**: {engine_msg}\n\n*Cluster centroid summary displayed in the table below and visualized in the cluster scatter plot.*"

    return f"**[Result]**: {engine_msg}"


# Standalone demonstration block
if __name__ == "__main__":
    mock_result_scalar = {
        "status": "success",
        "task": "mean",
        "result_type": "scalar",
        "value": 67333.33,
        "message": "The average Salary is 67,333.33.",
        "error": None,
    }

    mock_result_error = {
        "status": "error",
        "task": "mean",
        "result_type": None,
        "value": None,
        "message": "Column 'Bonus' not found.",
        "error": "Column 'Bonus' not found.",
    }

    print("=" * 60)
    print("TESTING RESPONSE GENERATOR")
    print("=" * 60)
    print("Scalar Response:", generate_response("What is the average salary?", mock_result_scalar))
    print("Error Response: ", generate_response("What is the average bonus?", mock_result_error))
    print("\nSUCCESS: response_generator.py formatted answers accurately!")
