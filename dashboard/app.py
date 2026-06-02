import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")


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
    This app benchmarks different configurations of my PolicyGPT RAG backend on a fixed golden dataset of questions.  
    It shows retrieval **recall**, **faithfulness**, **correctness**, and **latency** for each run, and lets you compare runs side by side and inspect failures.

    **Why it matters**  
    Instead of guessing whether a new retrieval strategy or model is better, I can now run a batch eval in minutes and see exactly how it changes quality and latency.

    **Example outcome**  
    In my first experiments, hybrid retrieval (BM25 + dense) improved context recall and faithfulness over dense-only retrieval, with a small latency cost.  
    That evidence is what I used to choose the default config in the main PolicyGPT app.
    """
        )

    st.markdown(
            """
    **Why this exists:**  
    It’s hard to know whether a new retrieval or model configuration actually improves a RAG system.  
    This lab benchmarks different PolicyGPT configs on a fixed golden set and shows their **recall, faithfulness, correctness, and latency** side by side.
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

    compare_df = None
    if compare_label != "(none)":
        compare_df = load_run(files[file_labels.index(compare_label)])

    st.subheader("Primary run summary")
    st.write(primary_df[["run_name", "model_name", "retrieval_strategy", "top_k"]].head(1))

    agg_primary = primary_df[["recall", "faithfulness", "correctness", "latency_ms"]].agg(
        ["mean", "std"]
    )
    st.dataframe(agg_primary.style.format("{:.3f}"))

    # Side-by-side comparison (aggregate)
    if compare_df is not None:
        st.subheader("Run comparison (aggregate)")

        def summarize(df: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "recall_mean": df["recall"].mean(),
                    "faithfulness_mean": df["faithfulness"].mean(),
                    "correctness_mean": df["correctness"].mean(),
                    "latency_ms_mean": df["latency_ms"].mean(),
                    "retrieval_strategy": df["retrieval_strategy"].iloc[0],
                    "model_name": df["model_name"].iloc[0],
                }
            )

        comp_df = pd.DataFrame(
            {
                primary_label: summarize(primary_df),
                compare_label: summarize(compare_df),
            }
        ).T

        # Select numeric columns only
        numeric_cols = comp_df.select_dtypes(include="number").columns

        st.dataframe(
            comp_df.style.format("{:.3f}", subset=numeric_cols)
)    
    # --- Metric callouts for primary run ---
    st.subheader("Primary run at a glance")

    avg_recall = primary_df["recall"].mean()
    avg_faith = primary_df["faithfulness"].mean()
    avg_corr = primary_df["correctness"].mean()
    avg_lat = primary_df["latency_ms"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg recall", f"{avg_recall:.2f}")
    c2.metric("Avg faithfulness", f"{avg_faith:.2f}")
    c3.metric("Avg correctness", f"{avg_corr:.2f}")
    c4.metric("Avg latency (ms)", f"{avg_lat:.0f}")

    st.subheader("Filter & inspect examples")

    # Simple failure filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_faith = st.slider("Min faithfulness", 0.0, 1.0, 0.0, 0.05)
    with col_f2:
        max_faith = st.slider("Max faithfulness", 0.0, 1.0, 1.0, 0.05)
    with col_f3:
        max_latency = st.number_input("Max latency (ms)", value=float(primary_df["latency_ms"].max()))

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
        f"**Metrics:** recall={row['recall']:.2f}, faithfulness={row['faithfulness']:.2f}, correctness={row['correctness']:.2f}, latency={row['latency_ms']:.1f} ms"
    )


if __name__ == "__main__":
    main()