"""
app.py
------
Main Gradio Web Application for AI Data Analyst Agent.
Integrates:
- File upload (CSV / Excel)
- Automated data profiling and schema summary
- Natural language query answering with Pandas
- Full KDD Data Preprocessing, Cleaning & Imputation (fill_missing, drop_missing, drop_duplicates, drop_column, normalize)
- Download Modified Data button right below the Ask button
- Dynamic visualization rendering (Matplotlib)
- Conversation history tracking
"""

import os
import re
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from data_loader import load_dataset, get_dataset_overview
from data_profiler import generate_full_profile
from query_planner import plan_query
from analysis_engine import execute_plan
from response_generator import generate_response
from visualization import create_auto_chart
from llm_client import check_api_status, call_llm

# Ensure exports directory exists for modified dataset downloads
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


# Global / Session State Helper
class SessionState:
    def __init__(self):
        self.df = None
        self.dataset_name = None
        self.conversation_history = []

    def reset(self):
        self.df = None
        self.dataset_name = None
        self.conversation_history = []


# -------------------------------------------------------------
# Gradio Event Handlers
# -------------------------------------------------------------

def on_file_upload(file_obj, current_history):
    """Handles dataset file upload, extracts profiling, and initializes preview."""
    if file_obj is None:
        return (
            None,  # preview_df
            "Please upload a CSV or Excel file to begin analysis.",  # status_markdown
            None,  # numerical_stats_df
            None,  # missing_summary_df
            [],    # chatbot reset
            None,  # result_table
            None,  # result_plot
            None,  # active_df
            "dataset",  # dataset_name_state
            None,  # btn_download
        )

    file_path = file_obj.name
    dataset_name = os.path.basename(file_path)

    df, err = load_dataset(file_path)
    if err:
        return (
            None,
            f"**Error Loading File**: {err}",
            None,
            None,
            [],
            None,
            None,
            None,
            dataset_name,
            None,
        )

    # Save initial export copy for downloading
    clean_name = os.path.splitext(dataset_name)[0].replace(" ", "_")
    export_file = os.path.join(EXPORTS_DIR, f"modified_{clean_name}.csv")
    df.to_csv(export_file, index=False)

    # Compute profiling summaries
    overview = get_dataset_overview(df, dataset_name=dataset_name)
    profile = generate_full_profile(df)

    status_text = (
        f"### Dataset: `{dataset_name}`\n"
        f"- **Rows**: {overview['total_rows']:,} | **Columns**: {overview['total_columns']}\n"
        f"- **Missing Cells**: {profile['total_missing_cells']} ({profile['overall_missing_percentage']}%)\n"
        f"- **Duplicate Rows**: {profile['duplicate_info']['duplicate_count']}\n"
        f"- **Numerical Columns**: {', '.join(profile['column_types']['numerical']) or 'None'}\n"
        f"- **Categorical Columns**: {', '.join(profile['column_types']['categorical']) or 'None'}"
    )

    # Dynamic sample questions based on actual dataset columns
    num_cols = profile["column_types"]["numerical"]
    cat_cols = profile["column_types"]["categorical"]
    valid_num_cols = [c for c in num_cols if not c.lower().endswith("id") and c.lower() != "id"]
    chosen_num = valid_num_cols[0] if valid_num_cols else (num_cols[0] if num_cols else None)

    if "Salary" in df.columns:
        example_q1 = "'What is the average salary?'"
        example_q2 = "'Show average salary by department'" if "Department" in df.columns else "'Check for missing values'"
    elif chosen_num and cat_cols:
        example_q1 = f"'What is the average {chosen_num}?'"
        example_q2 = f"'Create pie chart for {cat_cols[0]}' or 'Show average {chosen_num} by {cat_cols[0]}'"
    elif chosen_num:
        example_q1 = f"'What is the average {chosen_num}?'"
        example_q2 = f"'Plot histogram for {chosen_num}'"
    else:
        example_q1 = "'How many rows are in this dataset?'"
        example_q2 = "'Check for missing values'"

    welcome_chat = [
        {
            "role": "assistant",
            "content": (
                f"Hello! I have loaded `{dataset_name}` with {overview['total_rows']:,} rows and {overview['total_columns']} columns. "
                f"You can ask questions in English like {example_q1} or {example_q2}."
            ),
        }
    ]

    return (
        df.head(10),                             # preview_df
        status_text,                             # status_markdown
        profile["numerical_stats_table"],        # numerical_stats_df
        profile["missing_summary_table"],        # missing_summary_df
        welcome_chat,                            # chatbot
        None,                                    # result_table
        None,                                    # result_plot
        df,                                      # active_df
        dataset_name,                            # dataset_name_state
        export_file,                             # btn_download
    )


def _is_conversational_query(question: str) -> bool:
    """Detects whether a user message is a greeting or general conversational query."""
    q_clean = (question or "").strip().lower()
    greetings = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "sup", "yo", "who are you", "who are you?",
        "what can you do", "what can you do?", "help", "how to use", "how do i use this",
        "thanks", "thank you"
    }
    if q_clean in greetings:
        return True
    words = re.findall(r"\b\w+\b", q_clean)
    if len(words) <= 3 and any(w in greetings for w in words):
        analytical_keywords = {
            "average", "mean", "median", "sum", "count", "max", "min", "std", "chart",
            "plot", "histogram", "pie", "bar", "box", "scatter", "missing", "null",
            "filter", "predict", "train", "cluster", "drop", "fill"
        }
        if not any(w in words for w in analytical_keywords):
            return True
    return False


def _generate_conversational_reply(question: str, df: pd.DataFrame, dataset_name: str) -> str:
    """Uses Ollama (llama3.2) to generate a warm, conversational answer with dataset context."""
    status = check_api_status()
    if status.get("configured"):
        cols_summary = ", ".join(df.columns[:20]) if df is not None and not df.empty else "None"
        num_cols = ", ".join(df.select_dtypes(include=[np.number]).columns[:5]) if df is not None and not df.empty else "None"
        cat_cols = ", ".join(df.select_dtypes(include=["object", "category"]).columns[:5]) if df is not None and not df.empty else "None"
        total_rows = len(df) if df is not None else 0
        total_cols = len(df.columns) if df is not None else 0

        sys_prompt = (
            "You are ADAA (AI Data Analyst Agent), a helpful and expert AI data science copilot powered by Ollama (llama3.2).\n"
            f"Active dataset: '{dataset_name}' with {total_rows:,} rows and {total_cols} columns.\n"
            f"Columns: {cols_summary}.\n"
            f"Key numerical columns: {num_cols}. Key categorical columns: {cat_cols}.\n\n"
            "Instructions:\n"
            "1. If the user is greeting you (e.g. 'hi', 'hello') or asking who you are or what you can do, greet them warmly and introduce yourself as their AI Data Analyst Agent. Mention the dataset name and rows/columns, and suggest 3-4 concrete questions they can ask you to execute (e.g., statistics, data quality, charts like pie/bar charts, machine learning).\n"
            "2. If the user asks for data analysis advice or questions about the dataset, provide clear, concise insights and suggest next queries to execute.\n"
            "3. Keep the answer friendly, professional, concise, and formatted in clean markdown without any system error text."
        )
        reply, err = call_llm(question, system_instruction=sys_prompt, timeout=45)
        if not err and reply and reply.strip():
            return reply.strip()

    # Deterministic fallback if Ollama is unreachable
    col_sample = df.columns[1] if df is not None and len(df.columns) > 1 else (df.columns[0] if df is not None and len(df.columns) > 0 else "data")
    return (
        f"**Hello! I am your AI Data Analyst Agent.**\n\n"
        f"I have loaded **`{dataset_name}`** with **{len(df):,} rows** and **{len(df.columns)} columns**.\n\n"
        f"You can ask me questions to execute on this dataset, such as:\n"
        f"- *'What is the average {col_sample}?'*\n"
        f"- *'Create a pie chart for {col_sample}'*\n"
        f"- *'Show missing values across all columns'*\n"
        f"- *'Plot histogram for {col_sample}'*\n"
        f"- *'Train a Random Forest model'* or *'Show data quality report'*\n\n"
        f"What would you like to analyze?"
    )


def on_ask_question(question, active_df, dataset_name, chat_history, context_history):
    """Processes user questions via the LLM/Heuristic planner and analysis engine."""
    if not question or not question.strip():
        return (
            chat_history,
            context_history,
            None,
            None,
            "",
            active_df,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    if active_df is None or (isinstance(active_df, pd.DataFrame) and active_df.empty):
        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": "Please upload a CSV or Excel dataset first."})
        return (
            new_history,
            context_history,
            None,
            None,
            "",
            active_df,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    df = active_df
    dataset_name = dataset_name or "dataset"
    current_context = list(context_history or [])

    # 1. If user message is a greeting or general conversational query, reply via Ollama (llama3.2)
    if _is_conversational_query(question):
        reply = _generate_conversational_reply(question, df, dataset_name)
        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": reply})
        return (
            new_history,
            current_context,
            None,
            None,
            "",
            active_df,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # 2. Classify Query Intent with Scikit-Learn ML Model (Viva ML Component)
    from query_classifier import classify_user_query
    ml_intent, ml_conf = classify_user_query(question)

    # 3. Translate question into validated plan with context
    plan, plan_err = plan_query(question, df, conversation_history=current_context)
    if plan_err:
        # If the error is a column misspelling ("Did you mean: ..."), show notice
        if "did you mean" in plan_err.lower() or "not found" in plan_err.lower():
            err_reply = f"**[Notice]**: {plan_err}"
        else:
            # Answer conversationally using Ollama (llama3.2) with dataset context
            err_reply = _generate_conversational_reply(question, df, dataset_name)

        new_history = list(chat_history or [])
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": err_reply})
        return (
            new_history,
            current_context,
            None,
            None,
            "",
            active_df,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # 3. Execute plan with Pandas Analysis Engine
    result = execute_plan(df, plan)

    # 4. Record turn in contextual history for future follow-up questions
    target_column = plan.get("column") or plan.get("group_by") or plan.get("condition_col")
    current_context.append({
        "user": question,
        "task": plan.get("task"),
        "column": target_column,
        "analyzed_column": target_column,
    })

    # 5. Generate natural language response
    response_text = generate_response(question, result, dataset_name=dataset_name)

    new_history = list(chat_history or [])
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": response_text})

    # 6. Tabular result
    table_output = None
    if result.get("status") == "success" and isinstance(result.get("value"), pd.DataFrame):
        table_output = result["value"]

    # 7. Chart
    chart_fig = None
    if result.get("status") == "success":
        chart_fig, _ = create_auto_chart(result, df)

    # 8. Check if dataset was modified (KDD Preprocessing / Imputation / Cleaning / Transformation)
    if result.get("modified_df") is not None:
        updated_df = result["modified_df"]
        clean_name = os.path.splitext(dataset_name)[0].replace(" ", "_")
        export_file = os.path.join(EXPORTS_DIR, f"modified_{clean_name}.csv")
        updated_df.to_csv(export_file, index=False)

        overview = get_dataset_overview(updated_df, dataset_name=dataset_name)
        profile = generate_full_profile(updated_df)
        status_text = (
            f"### Dataset: `{dataset_name}` *(Modified)*\n"
            f"- **Rows**: {overview['total_rows']:,} | **Columns**: {overview['total_columns']}\n"
            f"- **Missing Cells**: {profile['total_missing_cells']} ({profile['overall_missing_percentage']}%)\n"
            f"- **Duplicate Rows**: {profile['duplicate_info']['duplicate_count']}\n"
            f"- **Numerical Columns**: {', '.join(profile['column_types']['numerical']) or 'None'}\n"
            f"- **Categorical Columns**: {', '.join(profile['column_types']['categorical']) or 'None'}"
        )
        return (
            new_history,
            current_context,
            table_output,
            chart_fig,
            "",
            updated_df,
            export_file,
            updated_df.head(10),
            status_text,
            profile["numerical_stats_table"],
            profile["missing_summary_table"],
        )

    return (
        new_history,
        current_context,
        table_output,
        chart_fig,
        "",
        active_df,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def on_load_sample_dataset():
    """Loads the pre-packaged sample dataset for instant demonstration."""
    sample_path = os.path.join("datasets", "sample_data.csv")
    if not os.path.exists(sample_path):
        return None, "Sample dataset not found.", None, None, [], None, None, None, "dataset", None

    class DummyFile:
        def __init__(self, path):
            self.name = path

    dummy = DummyFile(sample_path)
    return on_file_upload(dummy, [])


def on_clear():
    """Resets interface state."""
    plt.close("all")
    return (
        None,  # file upload
        None,  # preview df
        "Upload a CSV or Excel file to begin analysis.",  # status markdown
        None,  # numerical stats
        None,  # missing summary
        [],    # chat history
        None,  # result table
        None,  # result plot
        "",    # input box
        None,  # active_df
        "dataset",  # dataset_name_state
        None,  # btn_download
    )


# -------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------

api_status = check_api_status()
api_status_badge = (
    f"🟢 **LLM**: Connected (Ollama - {api_status['model']})"
    if api_status["configured"]
    else f"🟡 **Mode**: Heuristic Fallback ({api_status['message']})"
)

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.app-header {
    text-align: center;
    padding: 12px;
    background: linear-gradient(135deg, #1E3A8A, #3B82F6);
    color: white;
    border-radius: 8px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="AI Data Analyst Agent") as demo:
    gr.HTML(
        """
        <div class="app-header">
            <h1 style="margin:0; font-size: 26px;">🤖 AI Data Analyst Agent</h1>
            <p style="margin:4px 0 0 0; font-size: 14px; opacity: 0.9;">
                Natural-Language Conversational Tabular Analysis Powered by Pandas, Scikit-Learn & Ollama
            </p>
        </div>
        """
    )

    gr.Markdown(f"{api_status_badge} | **Supported Formats**: `.csv`, `.xlsx`, `.xls`")

    # Reactive states for session management
    active_df = gr.State(None)
    dataset_name_state = gr.State("dataset")
    context_history = gr.State([])

    with gr.Row():
        # LEFT COLUMN: File Upload & Profiling
        with gr.Column(scale=4):
            file_input = gr.File(
                label="📁 Upload Dataset (.csv or .xlsx)",
                file_types=[".csv", ".xlsx", ".xls"],
            )

            with gr.Row():
                btn_load_sample = gr.Button("📊 Load Sample Employee Dataset", variant="secondary")
                btn_clear = gr.Button("🔄 Reset Application", variant="stop")

            status_markdown = gr.Markdown("Upload a CSV or Excel file to begin analysis.")

            with gr.Accordion("📋 Dataset Profiling & Statistics", open=False):
                with gr.Tab("Data Preview"):
                    preview_df = gr.Dataframe(label="First 10 Rows", interactive=False)
                with gr.Tab("Numerical Statistics"):
                    numerical_stats_df = gr.Dataframe(label="Descriptive Statistics", interactive=False)
                with gr.Tab("Missing Values"):
                    missing_summary_df = gr.Dataframe(label="Missing Values Breakdown", interactive=False)

        # RIGHT COLUMN: Chat, Tables, and Charts
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                label="💬 Conversational Data Assistant",
                height=350,
            )

            # Query input and actions row (Ask button + Download button below)
            with gr.Row():
                query_input = gr.Textbox(
                    label="Ask a question about your data",
                    placeholder="e.g., Check missing values | Fill missing values in Age by mean | Correlation matrix",
                    scale=7,
                )
                with gr.Column(scale=3):
                    btn_send = gr.Button("Ask", variant="primary")
                    btn_download = gr.DownloadButton(
                        "📥 Download Modified Data",
                        variant="secondary",
                        size="sm",
                        visible=True,
                    )

            # Quick Question Suggestions - General Tasks & KDD Operations
            gr.Markdown("**🌐 Quick Prompts — General Tasks & KDD Operations:**")
            with gr.Row():
                btn_missing = gr.Button("🔍 Missing Values", size="sm")
                btn_corr = gr.Button("📈 Correlation Matrix", size="sm")
                btn_rows = gr.Button("🔢 Total Rows", size="sm")
                btn_cols = gr.Button("📊 Total Columns", size="sm")
                btn_dups = gr.Button("👥 Duplicate Rows", size="sm")
                btn_top = gr.Button("📋 Top 5 Records", size="sm")
                btn_outliers = gr.Button("⚠️ Detect Outliers", size="sm")
                btn_cluster = gr.Button("🎯 Cluster Data", size="sm")

            with gr.Accordion("💼 Sample Dataset Prompts (Employee Data)", open=False):
                with gr.Row():
                    q1 = gr.Button("Average Salary", size="sm")
                    q2 = gr.Button("Salary by Department", size="sm")
                    q3 = gr.Button("Top 3 Salaries", size="sm")
                    q4 = gr.Button("Salary > 70000", size="sm")

            with gr.Accordion("📊 Detailed Results & Charts", open=True):
                result_table = gr.Dataframe(label="Query Result Table", interactive=False)
                result_plot = gr.Plot(label="Generated Visualization")

    # -------------------------------------------------------------
    # UI Interactions & Event Wiring
    # -------------------------------------------------------------

    # File upload event
    file_input.change(
        fn=lambda f, c: on_file_upload(f, c) + ([],),
        inputs=[file_input, chatbot],
        outputs=[
            preview_df,
            status_markdown,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            active_df,
            dataset_name_state,
            btn_download,
            context_history,
        ],
    )

    # Load sample dataset button
    btn_load_sample.click(
        fn=lambda: on_load_sample_dataset() + ([],),
        inputs=[],
        outputs=[
            preview_df,
            status_markdown,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            active_df,
            dataset_name_state,
            btn_download,
            context_history,
        ],
    )

    # Question asking triggers
    ask_inputs = [query_input, active_df, dataset_name_state, chatbot, context_history]
    ask_outputs = [
        chatbot,
        context_history,
        result_table,
        result_plot,
        query_input,
        active_df,
        btn_download,
        preview_df,
        status_markdown,
        numerical_stats_df,
        missing_summary_df,
    ]

    btn_send.click(fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs)
    query_input.submit(fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs)

    # General task quick prompt buttons
    btn_missing.click(lambda: "Check for missing values in this dataset", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_corr.click(lambda: "Show correlation between numerical columns", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_rows.click(lambda: "How many rows are in this dataset?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_cols.click(lambda: "How many columns are in this dataset?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_dups.click(lambda: "Check for duplicate rows in this dataset", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_top.click(lambda: "Show the top 5 records", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_outliers.click(lambda: "Detect outliers in the numerical columns", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    btn_cluster.click(lambda: "Cluster records into 3 groups using KMeans", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )

    # Sample employee quick prompt buttons
    q1.click(lambda: "What is the average salary?", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    q2.click(lambda: "Show average salary by department", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    q3.click(lambda: "Top 3 employees by salary", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )
    q4.click(lambda: "Filter employees with salary greater than 70000", None, query_input).then(
        fn=on_ask_question, inputs=ask_inputs, outputs=ask_outputs
    )

    # Clear button (resets all visual components and context state)
    btn_clear.click(
        fn=lambda: on_clear() + ([],),
        inputs=[],
        outputs=[
            file_input,
            preview_df,
            status_markdown,
            numerical_stats_df,
            missing_summary_df,
            chatbot,
            result_table,
            result_plot,
            query_input,
            active_df,
            dataset_name_state,
            btn_download,
            context_history,
        ],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Data Analyst Agent Gradio Server...")
    print("=" * 60)
    # Launch Gradio server (automatically binds to 7860 or the next available port)
    demo.launch(server_name="127.0.0.1", share=False, css=custom_css)
