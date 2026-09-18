"""Mango leaf disease detector: side-by-side comparison of GourNet and GourNet v2."""

import pathlib
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from models import build_gournet, build_gournet_v2

APP_DIR = pathlib.Path(__file__).parent
WEIGHTS_DIR = APP_DIR / "weights"

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

MODEL_REGISTRY = {
    "GourNet (baseline)": {
        "short": "GourNet",
        "builder": build_gournet,
        "weights": {
            "Standard": WEIGHTS_DIR / "gournet_model.keras",
            "12k": WEIGHTS_DIR / "gournet_model_12k.keras",
        },
        "notes": "4-conv-block CNN.",
    },
    "GourNet v2 (enhanced)": {
        "short": "GourNet v2",
        "builder": build_gournet_v2,
        "weights": {
            "Standard": WEIGHTS_DIR / "gournet_v2_model.keras",
            "12k": WEIGHTS_DIR / "gournet_v2_model_12k.keras",
        },
        "notes": "GroupNorm + GAP variant.",
    },
}
MODEL_KEYS = list(MODEL_REGISTRY.keys())
WEIGHT_OPTIONS = ["Standard", "12k"]

st.set_page_config(
    page_title="GourNet model comparison",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_model(model_key: str, variant: str):
    """Load trained model for the selected weights variant, else build untrained architecture."""
    spec = MODEL_REGISTRY[model_key]
    fallback = spec["builder"](num_classes=len(CLASS_NAMES), input_shape=IMG_SIZE + (3,))
    weights_path = spec["weights"][variant]
    if weights_path.exists():
        try:
            return tf.keras.models.load_model(weights_path), True
        except Exception as e:
            st.warning(f"Could not load {weights_path.name} ({e}).")
            return fallback, False
    return fallback, False


def model_static_info(model, weights_path: pathlib.Path):
    try:
        params = int(model.count_params())
    except Exception:
        params = -1
    size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else 0.0
    return params, size_mb


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def timed_predict(model, batch: np.ndarray):
    t0 = time.perf_counter()
    probs = model.predict(batch, verbose=0)[0].astype(float)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return probs, dt_ms


def prediction_stats(probs: np.ndarray):
    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    conf1, conf2 = float(probs[top1]), float(probs[top2])
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log(clipped)))
    return {
        "top_idx": top1,
        "conf": conf1,
        "margin": conf1 - conf2,
        "entropy": entropy,
        "order": order,
    }


def guess_label_from_filename(filename: str) -> str:
    """Match a disease class name inside the file name, else Unknown."""
    stem = pathlib.Path(filename).stem.lower().replace("_", "").replace("-", "").replace(" ", "")
    for name in CLASS_NAMES:
        if name.lower().replace(" ", "") in stem:
            return name
    return "Unknown"


with st.sidebar:
    st.title("GourNet")
    variant = st.segmented_control("Weights", WEIGHT_OPTIONS, default="Standard")
    top_k = st.slider("Top-k classes per model", 1, 8, 3)

st.title("Mango leaf disease detector")
st.caption(f"Both models predict on the same uploaded image for direct comparison ({variant} weights).")

models, trained_flags, static_infos = {}, {}, {}
for key in MODEL_KEYS:
    m, ok = load_model(key, variant)
    models[key] = m
    trained_flags[key] = ok
    static_infos[key] = model_static_info(m, MODEL_REGISTRY[key]["weights"][variant])

a_key, b_key = MODEL_KEYS[0], MODEL_KEYS[1]

with st.container(horizontal=True):
    for key in MODEL_KEYS:
        params, size_mb = static_infos[key]
        ok = trained_flags[key]
        with st.container(border=True):
            st.markdown(f"**{key}**")
            st.caption(MODEL_REGISTRY[key]["notes"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Parameters", f"{params:,}" if params >= 0 else "n/a")
            c2.metric("File size", f"{size_mb:.1f} MB" if size_mb else "missing")
            c3.metric("Weights", "Trained" if ok else "Missing")

missing = [k for k in MODEL_KEYS if not trained_flags[k]]
if missing:
    st.warning(f"{variant} weights missing for: {', '.join(missing)}. Predictions are not valid.")
else:
    st.success(f"{variant} weights loaded for both models.")

uploaded = st.file_uploader("Upload a mango leaf photo", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a leaf image to run the comparison.")
else:
    img = Image.open(uploaded)
    batch = preprocess_image(img)

    st.image(img, caption="Uploaded image", width="content")

    results = {}
    with st.spinner("Running both models..."):
        for key in MODEL_KEYS:
            probs, dt_ms = timed_predict(models[key], batch)
            results[key] = {"probs": probs, "ms": dt_ms, "stats": prediction_stats(probs)}

    a, b = results[a_key], results[b_key]
    agree = a["stats"]["top_idx"] == b["stats"]["top_idx"]

    if agree:
        st.success(
            f"Both models predict **{CLASS_NAMES[a['stats']['top_idx']]}** — "
            f"{MODEL_REGISTRY[a_key]['short']} {a['stats']['conf'] * 100:.1f}% vs "
            f"{MODEL_REGISTRY[b_key]['short']} {b['stats']['conf'] * 100:.1f}%."
        )
    else:
        st.warning(
            f"Models disagree — **{MODEL_REGISTRY[a_key]['short']}**: "
            f"{CLASS_NAMES[a['stats']['top_idx']]} ({a['stats']['conf'] * 100:.1f}%) vs "
            f"**{MODEL_REGISTRY[b_key]['short']}**: "
            f"{CLASS_NAMES[b['stats']['top_idx']]} ({b['stats']['conf'] * 100:.1f}%). "
            "Review recommended."
        )

    col_a, col_b = st.columns(2)
    for col, key in zip((col_a, col_b), MODEL_KEYS):
        r = results[key]
        s = r["stats"]
        with col:
            with st.container(border=True):
                st.subheader(MODEL_REGISTRY[key]["short"])
                st.metric(
                    "Predicted class",
                    CLASS_NAMES[s["top_idx"]],
                    f"{s['conf'] * 100:.1f}% confidence",
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{r['ms']:.0f} ms")
                m2.metric("Margin", f"{s['margin'] * 100:.1f} pp")
                m3.metric("Entropy", f"{s['entropy']:.2f}")
                st.caption("Top classes")
                topk_rows = [
                    {
                        "Class": CLASS_NAMES[int(i)],
                        "Probability": f"{float(r['probs'][int(i)]) * 100:.2f}%",
                    }
                    for i in s["order"][:top_k]
                ]
                st.dataframe(pd.DataFrame(topk_rows), hide_index=True)

    with st.container(horizontal=True):
        st.metric(
            "Faster model",
            MODEL_REGISTRY[a_key if a["ms"] <= b["ms"] else b_key]["short"],
            f"{abs(a['ms'] - b['ms']):.0f} ms difference",
            border=True,
        )
        st.metric(
            "Higher confidence",
            MODEL_REGISTRY[a_key if a["stats"]["conf"] >= b["stats"]["conf"] else b_key]["short"],
            f"{abs(a['stats']['conf'] - b['stats']['conf']) * 100:.1f} pp difference",
            border=True,
        )
        st.metric(
            "Larger margin",
            MODEL_REGISTRY[a_key if a["stats"]["margin"] >= b["stats"]["margin"] else b_key]["short"],
            f"{abs(a['stats']['margin'] - b['stats']['margin']) * 100:.1f} pp difference",
            border=True,
        )

    st.subheader("Per-class probabilities")
    comp_df = pd.DataFrame(
        {
            MODEL_REGISTRY[a_key]["short"]: a["probs"],
            MODEL_REGISTRY[b_key]["short"]: b["probs"],
        },
        index=CLASS_NAMES,
    )
    st.bar_chart(comp_df, horizontal=True, stack=False, x_label="Probability", y_label="Disease class")

    st.subheader("Probability delta (v2 − baseline)")
    delta = b["probs"] - a["probs"]
    delta_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            MODEL_REGISTRY[a_key]["short"]: np.round(a["probs"], 4),
            MODEL_REGISTRY[b_key]["short"]: np.round(b["probs"], 4),
            "Delta": np.round(delta, 4),
        }
    ).sort_values("Delta", ascending=False)
    st.dataframe(
        delta_df,
        hide_index=True,
        column_config={
            MODEL_REGISTRY[a_key]["short"]: st.column_config.NumberColumn(format="%.4f"),
            MODEL_REGISTRY[b_key]["short"]: st.column_config.NumberColumn(format="%.4f"),
            "Delta": st.column_config.NumberColumn(format="+%.4f"),
        },
    )

    with st.expander("Raw probabilities"):
        for key in MODEL_KEYS:
            st.markdown(f"**{key}**")
            for i in results[key]["stats"]["order"]:
                st.write(f"{CLASS_NAMES[int(i)]}: {float(results[key]['probs'][int(i)]) * 100:.2f}%")

st.divider()

st.header("Batch benchmark")
st.caption("Upload multiple images to compare agreement rate, accuracy, and mean latency.")

multi = st.file_uploader(
    "Upload leaf photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="batch_uploader",
)

if multi:
    label_rows = pd.DataFrame(
        [{"File": f.name, "True label": guess_label_from_filename(f.name)} for f in multi]
    )
    auto_count = int((label_rows["True label"] != "Unknown").sum())
    if auto_count:
        st.caption(
            f"Detected labels from file names for {auto_count}/{len(label_rows)} images. "
            "Name files like Anthracnose_01.jpg. Correct any mistakes below."
        )
    else:
        st.caption("Name files like Anthracnose_01.jpg to auto-fill labels, or set them below.")
    edited = st.data_editor(
        label_rows,
        key="batch_labels",
        hide_index=True,
        column_config={
            "File": st.column_config.TextColumn(disabled=True),
            "True label": st.column_config.SelectboxColumn(
                "True label",
                options=["Unknown"] + CLASS_NAMES,
                required=True,
            ),
        },
    )

    if st.button("Run batch benchmark", icon=":material/play_arrow:"):
        records = []
        bar = st.progress(0, text="Running batch inference...")
        for i, f in enumerate(multi):
            try:
                bimg = Image.open(f)
                bbatch = preprocess_image(bimg)
            except Exception as e:
                st.warning(f"Skipping {f.name}: {e}")
                continue
            row = {"File": f.name}
            for key in MODEL_KEYS:
                short = MODEL_REGISTRY[key]["short"]
                probs, dt_ms = timed_predict(models[key], bbatch)
                s = prediction_stats(probs)
                row[f"{short} pred"] = CLASS_NAMES[s["top_idx"]]
                row[f"{short} conf"] = round(s["conf"], 4)
                row[f"{short} ms"] = round(dt_ms, 1)
                row[f"{short} margin"] = round(s["margin"], 4)
                row[f"{short} entropy"] = round(s["entropy"], 4)
            records.append(row)
            bar.progress((i + 1) / len(multi), text=f"Processed {i + 1}/{len(multi)}")
        bar.empty()

        if records:
            res_df = pd.DataFrame(records)
            truth = {r["File"]: r["True label"] for r in edited.to_dict("records")}
            res_df["True label"] = res_df["File"].map(truth)
            res_df["Agree"] = res_df[f"{MODEL_REGISTRY[a_key]['short']} pred"] == res_df[
                f"{MODEL_REGISTRY[b_key]['short']} pred"
            ]

            scored = res_df[res_df["True label"] != "Unknown"]
            with st.container(horizontal=True):
                st.metric("Images", str(len(res_df)), border=True)
                st.metric("Agreement rate", f"{res_df['Agree'].mean() * 100:.1f}%", border=True)
                for key in MODEL_KEYS:
                    short = MODEL_REGISTRY[key]["short"]
                    st.metric(f"{short} mean latency", f"{res_df[f'{short} ms'].mean():.0f} ms", border=True)
            if len(scored):
                with st.container(horizontal=True):
                    for key in MODEL_KEYS:
                        short = MODEL_REGISTRY[key]["short"]
                        acc = (scored[f"{short} pred"] == scored["True label"]).mean() * 100
                        st.metric(f"{short} accuracy (n={len(scored)})", f"{acc:.1f}%", border=True)
            else:
                st.info("Assign true labels above to score accuracy.")

            st.dataframe(res_df, hide_index=True)
            disagreements = res_df[~res_df["Agree"]]
            if len(disagreements):
                st.subheader("Disagreements")
                st.dataframe(disagreements, hide_index=True)
        else:
            st.error("No images could be processed.")
