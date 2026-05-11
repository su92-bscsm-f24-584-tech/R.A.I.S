import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import time
from main import run_pipeline
from utility import COMPANY_MAP, resolve, recursive_lstm_predict

plt.style.use("dark_background")
sns.set_style("dark")

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Intelligence Terminal", layout="wide")

# =========================
# SESSION STATE
# =========================
if "stage" not in st.session_state:
    st.session_state.stage = "intro"

if "result" not in st.session_state:
    st.session_state.result = None

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None


# =========================
# SIDEBAR (ONLY AFTER GO)
# =========================

# =========================
# INTRO SCREEN
# =========================
if st.session_state.stage == "intro":

    st.title("🧠 Financial Intelligence Research Grade")
    
    placeholder = st.empty()

    placeholder.markdown("### 🧠 Loading AI Brain...")

    time.sleep(0.8)

    placeholder.markdown("### 🚀 AI Analysis!")
    try:
        with open("animation.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=400)
    except:
        st.warning("Animation not found")

    if st.button("🚀 GO"):
        st.session_state.stage = "select"
        st.rerun()


# =========================
# COMPANY SELECTION
# =========================
if st.session_state.stage == "select":

    st.subheader("🏢 Select Company")

    cols = st.columns(4)
    selected_ticker = None

    for i, (ticker, name) in enumerate(COMPANY_MAP.items()):
        with cols[i % 4]:
            if st.button(name, key=ticker):
                selected_ticker = ticker

    manual = st.text_input("Or type company name")

    if manual:
        selected_ticker = resolve(manual)

    if st.button("🚀 RUN ANALYSIS") and selected_ticker:
        st.session_state.selected_ticker = selected_ticker
        st.session_state.stage = "analysis"

        with st.spinner("Running AI Brain Pipeline..."):
            st.session_state.result = run_pipeline(selected_ticker)

        st.rerun()


# =========================
# DASHBOARD
# =========================
if st.session_state.stage == "analysis" and st.session_state.result:

    r = st.session_state.result

    if st.button("🔄 New Analysis"):
        st.session_state.stage = "select"
        st.session_state.result = None
        st.rerun()

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🧠 Technical Brain",
    "📰 Emotion AI",
    "📈 Future Intelligence",
    "🔬 Meta",
    "🔁 Recursive LSTM",
    "🏗️ Architecture"
])  # OVERVIEW
    # =========================
    with tab1:
        fusion = r["fusion"]
        score = fusion["final_score"]

        st.markdown("## 📊 Market Intelligence Overview")

        st.metric("Confidence Score", f"{score*100:.2f}%")
        st.progress(min(abs(score), 1.0))

        # =========================
        # BUILD COMPARISON METRICS
        # =========================
        lstm_score = r["lstm"]["confidence"]
        sentiment_score = r["sentiment"]["sentiment_signal"]
        meta_score = r["meta_score"]
        fusion_score = score

        labels = ["LSTM", "Sentiment", "Fusion", "Meta"]

        values = [
            lstm_score,
            sentiment_score,
            fusion_score,
            meta_score
        ]

        # normalize for visualization (IMPORTANT)
        values = [max(min(v, 1), -1) for v in values]

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(labels, values, marker="o", linewidth=2)
        ax.fill_between(labels, values, alpha=0.2)

        ax.set_title("AI Multi-Brain Consensus Map")
        ax.set_ylabel("Signal Strength (-1 to +1)")
        ax.set_ylim(-1, 1)

        st.pyplot(fig)
    with tab2:
        st.subheader("🧠 Technical AI Models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### LSTM Trend Output")

            st.json(r["lstm"])

            fig, ax = plt.subplots(figsize=(5, 3))  # LOWER GRAPH SIZE
            ax.bar(r["lstm"].keys(), r["lstm"].values())

            ax.set_title("LSTM Trend Strength")
            ax.set_xlabel("Indicators")
            ax.set_ylabel("Score")

            st.pyplot(fig)

        with col2:
            st.markdown("### Trend Flow Visualization")

            values = list(r["lstm"].values())

            fig2, ax2 = plt.subplots()
            ax2.plot(values, marker="o")

            ax2.set_title("Trend Movement Line")
            ax2.set_xlabel("Time Steps")
            ax2.set_ylabel("Trend Strength")

            st.pyplot(fig2)

    # =========================
    # EMOTION AI
    # =========================
    with tab3:
        st.subheader("📰 Emotion Intelligence Layer")

        sent = r["sentiment"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Positive", sent["counts"]["positive"])
        col2.metric("Neutral", sent["counts"]["neutral"])
        col3.metric("Negative", sent["counts"]["negative"])
        with st.expander("📥 Sentiment AI Input Texts"):
            for i, txt in enumerate(sent["sen_inputs"]):
                st.markdown(f"**News {i+1}:**")
                st.write(txt)
                st.markdown("---")

        fig, ax = plt.subplots()
        ax.bar(sent["counts"].keys(), sent["counts"].values())

        ax.set_title("Market Emotion Distribution")
        ax.set_xlabel("Emotion Type")
        ax.set_ylabel("Count")

        st.pyplot(fig)

    # =========================
    # PRICE INTELLIGENCE
    # =========================
    with tab4:
        st.subheader("📈 Price Intelligence Panel")

        forecast = r["prophet"]

        st.metric("Current Price ($)", forecast["current_price"])
        st.metric("Predicted Price ($)", forecast["future_price"])

        fig, ax = plt.subplots()

        ax.plot(forecast["forecast_series"]["ds"],
                forecast["forecast_series"]["yhat"])

        ax.set_title("Price Forecast Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")

        st.pyplot(fig)

    # =========================
    # META FUSION
    # =========================
    with tab5:
        st.subheader("🔬 Meta Intelligence Layer")

        fusion = r["fusion"]

        meta_score = float(r["meta_score"]) * 100

        st.metric("Meta Score", f"{meta_score:.2f}%")

        fig, ax = plt.subplots()

        keys = list(fusion.keys())
        vals = [abs(float(v)) if isinstance(v, (int, float)) else 0 for v in fusion.values()]

        ax.bar(keys, vals)

        ax.set_title("Meta Fusion Weights")
        ax.set_xlabel("Models")
        ax.set_ylabel("Influence Score")

        st.pyplot(fig)
    with tab6:

        st.subheader("🔁 Recursive LSTM Forecasting Engine")

        st.markdown("Simulated multi-step prediction using LSTM recursion")

        steps = st.slider("Prediction Steps", 5, 50, 10)

        if st.button("Run Recursive Forecast"):
            try:
                model = r.get("lstm_model", None)
                init_seq = r.get("init_seq", None)
                device = r.get("device", None)

                if model is None:
                    st.warning("Recursive model not available in pipeline output")
                else:
                    preds = recursive_lstm_predict(model, init_seq, steps, device)
                    st.line_chart(preds)

            except Exception as e:
                st.error(f"Recursive LSTM failed: {str(e)}")
    with tab7:
            
        st.subheader("🏗️ System Architecture")

        st.markdown("AI Multi-Model Financial Intelligence Pipeline")

        st.graphviz_chart("""
        digraph {
            User -> Resolver
            Resolver -> LSTM
            Resolver -> Sentiment
            Resolver -> Forecast

            LSTM -> Fusion
            Sentiment -> Fusion
            Forecast -> Fusion

            Fusion -> MetaModel
            MetaModel -> Dashboard
        }
        """)
        st.subheader("🏗️ System Architecture NON-TECHNICAL")

        st.image("mermaid.png", caption="Sunrise by the mountains")
