"""Real or AI? Human vs. Model - Gradio game (deployed on Render)."""
import os, json, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TF logs

import numpy as np
import gradio as gr
from PIL import Image
import keras

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "test_images")
N_ROUNDS = 10
LABEL_TXT = {1: "Real", 0: "AI-generated"}

# ---- Model & data (loaded once at startup) ----
model = keras.models.load_model(os.path.join(BASE, "best_model.keras"), compile=False)
with open(os.path.join(BASE, "labels.json")) as f:
    LABELS = json.load(f)                       # filename -> 1 (Real) / 0 (AI); never sent to browser
FILES = sorted(LABELS)


def preprocess(path):
    """Identical to notebook X_test_pp: RGB, float32, [0,255]. NO /255 - model rescales internally."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)[None]


def p_real(fname):
    """Model's live prediction: P(real) from the sigmoid output."""
    return float(model(preprocess(os.path.join(IMG_DIR, fname)), training=False).numpy()[0, 0])


p_real(FILES[0])                                # warm-up so the first click is fast


# ---- Game state & rendering ----
def new_state():
    return dict(order=random.sample(FILES, N_ROUNDS), round=1, human=0, model=0,
                answered=False, fb="<div class='fb'>👀 Real photo or AI-generated? Make your call.</div>")


def score_html(s):
    played = s["round"] if s["answered"] else s["round"] - 1
    def card(who, pts, color):
        acc = f"{pts / played:.0%}" if played else "–"
        return (f"<div class='card' style='border-color:{color}'><div class='who'>{who}</div>"
                f"<div class='pts'>{pts}</div><div class='acc'>accuracy {acc}</div></div>")
    return f"<div class='board'>{card('🧑 You', s['human'], '#3b82f6')}{card('🤖 Model', s['model'], '#a855f7')}</div>"


def feedback_html(truth, choice, m_pred, p):
    h_ok, m_ok = choice == truth, m_pred == truth
    conf = p if m_pred == 1 else 1 - p
    color = "#16a34a" if h_ok else "#dc2626"
    return (f"<div class='fb' style='border-left:6px solid {color}'>"
            f"<b>Answer: {LABEL_TXT[truth]}</b><br>"
            f"{'✅' if h_ok else '❌'} You said <b>{LABEL_TXT[choice]}</b><br>"
            f"{'✅' if m_ok else '❌'} Model said <b>{LABEL_TXT[m_pred]}</b> "
            f"({conf:.0%} confident · P(real) = {p:.2f})</div>")


def final_html(s):
    h, m = s["human"], s["model"]
    msg = ("🎉 You beat the model! 🎉" if h > m else
           "🤖 The model wins this time." if h < m else "🤝 It's a tie!")
    return (f"<div class='final'><h2>{msg}</h2>You: <b>{h}/{N_ROUNDS}</b> · Model: <b>{m}/{N_ROUNDS}</b><br>"
            f"Press <b>Play again</b> for a new set of faces.</div>")


def render(s):
    a, last = s["answered"], s["round"] == N_ROUNDS
    return (s,
            os.path.join(IMG_DIR, s["order"][s["round"] - 1]),
            f"### Round {s['round']} / {N_ROUNDS}",
            score_html(s),
            s["fb"],
            gr.Button(interactive=not a),             # Real
            gr.Button(interactive=not a),             # AI
            gr.Button(interactive=a and not last))    # Next


def start():
    return render(new_state())


def guess(choice, s):
    if s["answered"]:                               # ignore double clicks
        return render(s)
    fname = s["order"][s["round"] - 1]
    truth, p = LABELS[fname], p_real(fname)
    m_pred = int(p >= 0.5)
    s["human"] += int(choice == truth)
    s["model"] += int(m_pred == truth)
    s["answered"] = True
    s["fb"] = feedback_html(truth, choice, m_pred, p)
    if s["round"] == N_ROUNDS:
        s["fb"] += final_html(s)
    return render(s)


def next_round(s):
    if s["answered"] and s["round"] < N_ROUNDS:
        s["round"] += 1
        s["answered"] = False
        s["fb"] = "<div class='fb'>👀 Real photo or AI-generated?</div>"
    return render(s)


# ---- UI ----
CSS = """
.board{display:flex;gap:12px}
.card{flex:1;border:2px solid;border-radius:12px;padding:10px;text-align:center}
.who{font-weight:600}.pts{font-size:2rem;font-weight:700;line-height:1.2}.acc{font-size:.85rem;opacity:.75}
.fb{padding:10px 14px;border-radius:8px;background:var(--block-background-fill);line-height:1.7}
.final{margin-top:12px;padding:14px;border-radius:12px;text-align:center;
       background:var(--block-background-fill);border:2px dashed var(--border-color-primary)}
"""

with gr.Blocks(title="Real or AI? Human vs. Model") as demo:
    state = gr.State()
    gr.Markdown(
        "# 🕵️ Real or AI? Human vs. Model\n"
        f"Look at each face and decide whether it's a **real photo** or **AI-generated**. "
        f"After you answer, the model reveals its guess. Play {N_ROUNDS} rounds and see who wins!")
    with gr.Row():
        with gr.Column(scale=3, min_width=300):
            image = gr.Image(type="filepath", interactive=False, show_label=False, height=380)
        with gr.Column(scale=2, min_width=300):
            round_box = gr.Markdown()
            score_box = gr.HTML()
            with gr.Row():
                btn_real = gr.Button("🧑 Real", variant="primary")
                btn_ai = gr.Button("🤖 AI-generated", variant="primary")
            feedback_box = gr.HTML()
            with gr.Row():
                btn_next = gr.Button("Next image ▶")
                btn_restart = gr.Button("🔄 Play again", variant="secondary")
    gr.Markdown(
        "<small>Model: MobileNetV3Small (ImageNet-pretrained), last 3 blocks fine-tuned with dropout 0.5 "
        "and label smoothing · 74.0% accuracy on the held-out test set · images are from the test set "
        "and were never used for training.</small>")

    OUT = [state, image, round_box, score_box, feedback_box, btn_real, btn_ai, btn_next]
    demo.load(start, outputs=OUT)
    btn_real.click(lambda s: guess(1, s), inputs=state, outputs=OUT)
    btn_ai.click(lambda s: guess(0, s), inputs=state, outputs=OUT)
    btn_next.click(next_round, inputs=state, outputs=OUT)
    btn_restart.click(start, outputs=OUT)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",                       # Render: must bind 0.0.0.0
                server_port=int(os.environ.get("PORT", 7860)),  # Render injects $PORT
                theme=gr.themes.Soft(), css=CSS)              # Gradio 6: theme/css go in launch()
