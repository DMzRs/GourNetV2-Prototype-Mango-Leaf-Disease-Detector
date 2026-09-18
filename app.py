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


def _iter_conv_layers(layer):
    for sub in getattr(layer, "layers", []):
        yield from _iter_conv_layers(sub)
    if isinstance(layer, tf.keras.layers.Conv2D):
        yield layer


def gradcam_cam(model, batch: np.ndarray) -> np.ndarray:
    """Normalized Grad-CAM map of the predicted class at model input resolution."""
    conv_layers = list(_iter_conv_layers(model))
    conv_layer = conv_layers[-1]
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch, training=False)
        top = int(tf.argmax(preds[0]))
        loss = preds[:, top]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0.0)
    cam = cam / (cam.max() + 1e-8)
    return np.array(Image.fromarray((cam * 255).astype("uint8")).resize(IMG_SIZE, Image.BILINEAR)).astype(float) / 255.0


def blend_heatmap(orig_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a normalized CAM over the photo with a JET-style colormap."""
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0, 1)
    base = np.array(orig_img.convert("RGB").resize(IMG_SIZE)).astype(float) / 255.0
    overlay = (1.0 - alpha) * base + alpha * np.stack([r, g, b], axis=-1)
    return Image.fromarray((np.clip(overlay, 0, 1) * 255).astype("uint8"))


def gradcam_overlay(model, batch: np.ndarray, orig_img: Image.Image, alpha: float = 0.45):
    """Grad-CAM heatmap of the predicted class blended over the leaf photo."""
    return blend_heatmap(orig_img, gradcam_cam(model, batch), alpha)


def cam_focus(cam: np.ndarray, frac: float = 0.10) -> float:
    """Share of heatmap mass in the hottest pixels. Higher means more focal, not more correct."""
    flat = cam.flatten()
    k = max(1, int(len(flat) * frac))
    return float(np.partition(flat, -k)[-k:].sum() / (flat.sum() + 1e-8))


def tta_stability(model, img: Image.Image, clean_top: int, n: int, seed: int = 7):
    """Repeat prediction on lightly augmented copies. Returns (stability, mean conf, std conf)."""
    rng = np.random.default_rng(seed)
    agree, confs = 0, []
    for _ in range(n):
        aug = img.convert("RGB")
        if rng.random() < 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        aug = aug.rotate(float(rng.uniform(-15, 15)), resample=Image.BILINEAR)
        factor = float(rng.uniform(0.9, 1.1))
        aug = Image.fromarray(np.clip(np.array(aug).astype(float) * factor, 0, 255).astype("uint8"))
        p = model.predict(preprocess_image(aug), verbose=0)[0].astype(float)
        t = int(np.argmax(p))
        confs.append(float(p[t]))
        if t == clean_top:
            agree += 1
    return agree / n, float(np.mean(confs)), float(np.std(confs))


with st.sidebar:
    st.title("GourNet")
    variant = st.segmented_control("Weights", WEIGHT_OPTIONS, default="Standard")
    top_k = st.slider("Top-k classes per model", 1, 8, 3)
    show_cam = st.toggle("Show Grad-CAM", value=True)
    run_tta = st.toggle("Run TTA stability check", value=True)
    tta_n = st.slider("TTA repeats per model", 5, 30, 10)

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

    with st.spinner("Computing Grad-CAM and TTA stability..."):
        for key in MODEL_KEYS:
            try:
                cam = gradcam_cam(models[key], batch)
                results[key]["cam"] = cam
                results[key]["focus"] = cam_focus(cam)
                results[key]["overlay"] = blend_heatmap(img, cam)
            except Exception as e:
                results[key]["cam_error"] = str(e)
            if run_tta:
                stab, mean_c, std_c = tta_stability(models[key], img, results[key]["stats"]["top_idx"], tta_n)
                results[key]["tta"] = (stab, mean_c, std_c)

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
                if show_cam:
                    if "overlay" in r:
                        st.image(r["overlay"], caption=f"Grad-CAM: {CLASS_NAMES[s['top_idx']]}")
                    else:
                        st.caption(f"Grad-CAM unavailable ({r.get('cam_error', 'error')}).")

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

    st.subheader("Robustness")
    st.caption("Focal and stable do not mean correct. Confirm against test-set accuracy.")
    with st.container(horizontal=True):
        if "focus" in a and "focus" in b:
            st.metric(
                "More focal CAM",
                MODEL_REGISTRY[a_key if a["focus"] >= b["focus"] else b_key]["short"],
                f"{max(a['focus'], b['focus']) * 100:.0f}% mass in hottest 10% area",
                border=True,
            )
        if run_tta and "tta" in a and "tta" in b:
            st.metric(
                "More stable (TTA)",
                MODEL_REGISTRY[a_key if a["tta"][0] >= b["tta"][0] else b_key]["short"],
                f"{max(a['tta'][0], b['tta'][0]) * 100:.0f}% agree over {tta_n} runs",
                border=True,
            )
    if run_tta and "tta" in a and "tta" in b:
        st.caption(
            f"TTA detail — {MODEL_REGISTRY[a_key]['short']}: {a['tta'][0] * 100:.0f}% stable, "
            f"conf {a['tta'][1] * 100:.1f}±{a['tta'][2] * 100:.1f}%; "
            f"{MODEL_REGISTRY[b_key]['short']}: {b['tta'][0] * 100:.0f}% stable, "
            f"conf {b['tta'][1] * 100:.1f}±{b['tta'][2] * 100:.1f}% "
            f"(flip, ±15° rotation, brightness over {tta_n} runs)."
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
