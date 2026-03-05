import gradio as gr
from transformers import pipeline

# --- THE BRAIN ---
# We use a fast Transformer model that detects the "intent" of your text.
print("Initializing Transformer Brain... Please wait.")
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

# --- THE HUMAN DEMO DATA ---
REACTIONS = {
    "Pain": {
        "visual": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXZueXJueXJueXJueXJueXJueXJueXJueXJueXJueXJueCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKVUn7iM8FMEU24/giphy.gif",
        "bio_log": "Nociceptors firing: Signal sent to Thalamus. Facial wince detected."
    },
    "Laughter": {
        "visual": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXZueXJueXJueXJueXJueXJueXJueXJueXJueXJueXJueCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l3fQf1OEAq0iri9RC/giphy.gif",
        "bio_log": "Dopamine spike: Involuntary contraction of zygomatic muscles."
    },
    "Surprise": {
        "visual": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXZueXJueXJueXJueXJueXJueXJueXJueXJueXJueXJueCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26ufdipQqU2lhNA4g/giphy.gif",
        "bio_log": "Amygdala alert: Immediate eyelid elevation and intake of breath."
    },
    "Anger": {
        "visual": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXZueXJueXJueXJueXJueXJueXJueXJueXJueXJueXJueCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o72F8t9TDi2xVnxOE/giphy.gif",
        "bio_log": "Adrenaline surge: Corrugator muscles drawing eyebrows together."
    },
    "Neutral": {
        "visual": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXZueXJueXJueXJueXJueXJueXJueXJueXJueXJueXJueCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bef5Ix9xW7mPy/giphy.gif",
        "bio_log": "Baseline state: No significant stimulus detected."
    }
}


def simulate_human(text):
    if not text.strip():
        return REACTIONS["Neutral"]["visual"], "Waiting for input...", "Status: IDLE"

    # The Transformer "Senses" the text
    labels = list(REACTIONS.keys())
    result = classifier(text, candidate_labels=labels)

    top_label = result['labels'][0]
    confidence = result['scores'][0]

    # Get reaction data
    data = REACTIONS[top_label]

    return data[
        "visual"], f"### Sensation: {top_label.upper()}", f"**Neural Log:** {data['bio_log']} \n\n*Confidence: {int(confidence * 100)}%*"


# --- THE UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Transformer Human-Sensation Demo")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Enter Action (e.g., 'I poke you')", placeholder="Type here...")
            btn = gr.Button("Apply Stimulus", variant="primary")
            log = gr.Markdown("### Neural Log\nWaiting for stimulus...")

        with gr.Column():
            display = gr.Image(label="Human Demo Face", value=REACTIONS["Neutral"]["visual"])
            label = gr.Markdown("### Sensation: NEUTRAL")

    btn.click(simulate_human, inputs=inp, outputs=[display, label, log])

if __name__ == "__main__":
    demo.launch()