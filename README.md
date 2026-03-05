
# 🧠 Human Sensation Simulator (Transformer-Based)

This project is a **Text-to-Behavior** demonstration that uses a state-of-the-art Transformer model to simulate human physical reactions. When a user inputs a text-based stimulus (like "I poke you with a needle"), the AI "senses" the intent and triggers a corresponding facial reaction from a single human subject.

## 🚀 How It Works

The system bridges the gap between **Natural Language Processing (NLP)** and **Visual Behavioral Modeling** through three main stages:

1. **Stimulus Input:** The user provides a text description of a physical action.
2. **Transformer Processing (The "Brain"):** The system uses a **BART-Large-MNLI** model. It utilizes the **Attention Mechanism** to calculate the semantic similarity between the user's text and biological sensations (Pain, Laughter, Surprise, etc.).
3. **Human Demo (The "Body"):** Based on the model's output, a specific facial expression is triggered in the UI, simulating a real-time biological response.

---

## 🛠️ Tech Stack

* **Model Architecture:** Transformer (Zero-Shot Classification)
* **Library:** `Hugging Face Transformers`
* **UI Framework:** `Gradio`
* **Language:** `Python 3.9+`
* **Backend:** `PyTorch`

---

## 📦 Installation & Local Setup

To run this project in **PyCharm** or your local terminal:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/human-sensation-transformer.git
cd human-sensation-transformer

```

### 2. Install Dependencies

```bash
pip install transformers torch gradio

```

### 3. Run the Application

```bash
python human_simulator.py

```

*The app will be available at `http://127.0.0.1:7860*`

---

## 📂 Project Structure

* `app.py`: The main script containing the Transformer logic and Gradio UI.
* `requirements.txt`: List of Python dependencies for deployment.
* **Neural Log:** The UI includes a "Brain Log" that explains the biological processing (e.g., Nociceptor activation for pain).

---

## 🧪 Example Stimuli to Try

| Input Text | Predicted Sensation | Biological Logic |
| --- | --- | --- |
| "I poke you with a sharp needle" | **PAIN** | Nociceptors triggered / Reflex arc active. |
| "I tell you a very funny joke" | **LAUGHTER** | Dopamine release / Zygomatic muscle contraction. |
| "I jump out and scare you!" | **SURPRISE** | Amygdala alert / Startle reflex. |
| "The room is quiet and calm" | **NEUTRAL** | Baseline state / System Idle. |

---

## 🌐 Deployment

This project is ready to be deployed to **Hugging Face Spaces**. Simply upload the `app.py` and `requirements.txt` to a new Gradio Space.

