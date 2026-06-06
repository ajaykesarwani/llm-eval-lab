import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")


def infer_task_type(df: pd.DataFrame) -> str:
    if {"faithfulness", "correctness"}.issubset(df.columns):
        return "qa"
    if {"bleu", "rouge_l"}.issubset(df.columns):
        return "summarization"
    return "unknown"


def list_result_files():
    p = Path(RESULTS_DIR)
    if not p.exists():
        return []
    return sorted(p.glob("*.csv"))


def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["run_file"] = path.name
    return df


def main():
    st.set_page_config(page_title="LLM Eval Lab – PolicyGPT", layout="wide")

    st.title("LLM Eval Lab for PolicyGPT")

    st.markdown(
        """
**What this tool does**  
This app benchmarks different configurations of my PolicyGPT RAG backend on fixed golden datasets.  
It shows retrieval **recall**, **faithfulness/correctness** for QA, **BLEU/ROUGE-L** for summarization, plus **latency** and **cost** for each run.
"""
    )

    st.markdown(
        """
**Why this exists**  
It’s hard to know whether a new retrieval or model configuration actually improves a RAG system.  
This lab benchmarks different PolicyGPT configs and shows their metrics side by side.
"""
    )

    files = list_result_files()
    if not files:
        st.warning(f"No result CSVs found in `{RESULTS_DIR}`. Run `eval_runner` first.")
        return

    st.sidebar.header("Run selection")

    file_labels = [f.name for f in files]
    primary_label = st.sidebar.selectbox("Primary run", file_labels, index=0)
    compare_label = st.sidebar.selectbox(
        "Compare with (optional)", ["(none)"] + file_labels, index=0
    )

    primary_df = load_run(files[file_labels.index(primary_label)])
    task_type = infer_task_type(primary_df)

    compare_df = None
    if compare_label != "(none)":
        compare_df = load_run(files[file_labels.index(compare_label)])

    st.subheader(f"Primary run summary ({task_type})")
    st.write(
        primary_df[
            ["run_name", "model_name", "retrieval_strategy", "top_k"]
        ].head(1)
    )

    # ---------- Aggregate metrics for primary run ----------
    if task_type == "qa":
        base_cols = ["recall", "faithfulness", "correctness", "latency_ms"]
    elif task_type == "summarization":
        base_cols = ["recall", "bleu", "rouge_l", "latency_ms"]
    else:
        base_cols = [c for c in ["recall", "latency_ms"] if c in primary_df.columns]

    existing_base = [c for c in base_cols if c in primary_df.columns]
    agg_df = primary_df[existing_base].agg(["mean", "std"])

    # Add cost/tokens if available
    if "cost_usd" in primary_df.columns:
        agg_df.loc["mean", "cost_usd"] = primary_df["cost_usd"].mean()
        agg_df.loc["std", "cost_usd"] = primary_df["cost_usd"].std()
    if "total_tokens" in primary_df.columns:
        agg_df.loc["mean", "total_tokens"] = primary_df["total_tokens"].mean()
        agg_df.loc["std", "total_tokens"] = primary_df["total_tokens"].std()

    st.dataframe(agg_df.style.format("{:.3f}"))

    # ---------- Side-by-side comparison (aggregate) ----------
    if compare_df is not None:
        st.subheader("Run comparison (aggregate)")

        def summarize_qa(df: pd.DataFrame) -> pd.Series:
            s = {
                "recall_mean": df["recall"].mean(),
                "faithfulness_mean": df["faithfulness"].mean(),
                "correctness_mean": df["correctness"].mean(),
                "latency_ms_mean": df["latency_ms"].mean(),
                "retrieval_strategy": df["retrieval_strategy"].iloc[0],
                "model_name": df["model_name"].iloc[0],
            }
            if "cost_usd" in df.columns:
                s["cost_usd_total"] = df["cost_usd"].sum()
                s["cost_usd_mean"] = df["cost_usd"].mean()
            return pd.Series(s)

        def summarize_summ(df: pd.DataFrame) -> pd.Series:
            s = {
                "recall_mean": df["recall"].mean() if "recall" in df.columns else 0.0,
                "bleu_mean": df["bleu"].mean() if "bleu" in df.columns else 0.0,
                "rouge_l_mean": df["rouge_l"].mean() if "rouge_l" in df.columns else 0.0,
                "latency_ms_mean": df["latency_ms"].mean() if "latency_ms" in df.columns else 0.0,
                "retrieval_strategy": df["retrieval_strategy"].iloc[0]
                if "retrieval_strategy" in df.columns
                else "",
                "model_name": df["model_name"].iloc[0] if "model_name" in df.columns else "",
            }
            if "cost_usd" in df.columns:
                s["cost_usd_total"] = df["cost_usd"].sum()
                s["cost_usd_mean"] = df["cost_usd"].mean()
            return pd.Series(s)

        if task_type == "qa":
            comp_df = pd.DataFrame(
                {
                    primary_label: summarize_qa(primary_df),
                    compare_label: summarize_qa(compare_df),
                }
            ).T
        elif task_type == "summarization":
            comp_df = pd.DataFrame(
                {
                    primary_label: summarize_summ(primary_df),
                    compare_label: summarize_summ(compare_df),
                }
            ).T
        else:
            comp_df = pd.DataFrame()

        if not comp_df.empty:
            numeric_cols = comp_df.select_dtypes(include="number").columns
            st.dataframe(
                comp_df.style.format("{:.3f}", subset=numeric_cols)
            )

    # ---------- Metric callouts for primary run ----------
    st.subheader("Primary run at a glance")

    avg_recall = primary_df["recall"].mean() if "recall" in primary_df.columns else 0.0
    avg_lat = primary_df["latency_ms"].mean()
    total_cost = primary_df["cost_usd"].sum() if "cost_usd" in primary_df.columns else 0.0
    avg_cost = primary_df["cost_usd"].mean() if "cost_usd" in primary_df.columns else 0.0

    if task_type == "qa":
        avg_faith = primary_df["faithfulness"].mean()
        avg_corr = primary_df["correctness"].mean()
        faith_label = "Avg faithfulness"
        corr_label = "Avg correctness"
    else:
        avg_faith = primary_df["rouge_l"].mean() if "rouge_l" in primary_df.columns else 0.0
        avg_corr = primary_df["bleu"].mean() if "bleu" in primary_df.columns else 0.0
        faith_label = "Avg ROUGE-L"
        corr_label = "Avg BLEU"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg recall", f"{avg_recall:.2f}")
    c2.metric(faith_label, f"{avg_faith:.3f}")
    # c3.metric(corr_label, f"{avg_corr:.3f if task_type=='qa' else avg_corr:.2f}")
    if task_type == "qa":
        c3.metric(corr_label, f"{avg_corr:.3f}")
    else:
        c3.metric(corr_label, f"{avg_corr:.2f}")
    c4.metric("Avg latency (ms)", f"{avg_lat:.0f}")

    c5, c6 = st.columns(2)
    c5.metric("Total cost (USD)", f"{total_cost:.4f}")
    c6.metric("Avg cost/query (USD)", f"{avg_cost:.4f}")

    # ---------- Detailed views ----------
    if task_type == "qa":
        st.subheader("Filter & inspect examples")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_faith = st.slider("Min faithfulness", 0.0, 1.0, 0.0, 0.05)
        with col_f2:
            max_faith = st.slider("Max faithfulness", 0.0, 1.0, 1.0, 0.05)
        with col_f3:
            max_latency = st.number_input(
                "Max latency (ms)",
                value=float(primary_df["latency_ms"].max()),
            )

        filtered = primary_df[
            (primary_df["faithfulness"] >= min_faith)
            & (primary_df["faithfulness"] <= max_faith)
            & (primary_df["latency_ms"] <= max_latency)
        ]

        st.markdown(f"Showing **{len(filtered)}** examples after filtering.")
        st.dataframe(
            filtered[
                [
                    "example_id",
                    "question",
                    "latency_ms",
                    "recall",
                    "faithfulness",
                    "correctness",
                ]
            ]
        )

        st.subheader("Metric distributions (primary run)")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(primary_df[["recall", "faithfulness", "correctness"]])
        with col2:
            st.bar_chart(primary_df[["latency_ms"]])

        st.subheader("Inspect individual example")
        example_ids = filtered["example_id"].tolist()
        if not example_ids:
            st.info("No examples in current filter; widen your filters to see details.")
            return

        ex_id = st.selectbox("Example ID", example_ids)
        row = filtered[filtered["example_id"] == ex_id].iloc[0]

        st.markdown(f"**Question:** {row['question']}")
        st.markdown(f"**Gold answer:** {row['gold_answer']}")
        st.markdown(f"**Model answer:** {row['answer']}")
        st.markdown(
            f"**Metrics:** recall={row['recall']:.2f}, "
            f"faithfulness={row['faithfulness']:.2f}, "
            f"correctness={row['correctness']:.2f}, "
            f"latency={row['latency_ms']:.1f} ms, "
            f"cost={row.get('cost_usd', 0.0):.4f} USD"
        )

    elif task_type == "summarization":
        st.subheader("Summarization examples")

        st.dataframe(
            primary_df[
                [
                    "example_id",
                    "source_text",
                    "gold_summary",
                    "summary",
                    "bleu",
                    "rouge_l",
                    "latency_ms",
                    "cost_usd",
                ]
            ]
        )

        st.subheader("Metric distributions (primary run)")
        col1, col2 = st.columns(2)
        with col1:
            if "bleu" in primary_df.columns:
                st.bar_chart(primary_df[["bleu"]])
        with col2:
            if "rouge_l" in primary_df.columns:
                st.bar_chart(primary_df[["rouge_l"]])

    else:
        st.info("Unknown task type for this run; showing numeric aggregates only.")


if __name__ == "__main__":
    main()