# 🏥 Real-Time Post-Surgery Follow-Up Voice Agent

This project implements a **real-time conversational medical voice agent** that performs a post-surgery recovery check-in with a patient who recently had ankle surgery. The agent listens, understands responses, detects emotional tone, and replies empathetically — all running on **CPU-only with low latency**.

The system is built on the **LiveKit Realtime AI framework**, supporting streaming ASR, LLM conversation orchestration, and neural text-to-speech.

A **demo video is included in the submission** to illustrate the live interaction.

---

## 🎬 Conversation Scenario

During the follow-up call, the agent:

1️⃣ Introduces the purpose of the call
2️⃣ Asks how the patient is feeling
3️⃣ Requests a pain score (1–10)
4️⃣ Guides the patient through three recovery exercises

* Ankle Mobility Stretch
* Toe Tapping
* Calf Raises
  and offers up to **1-minute pause time**
  5️⃣ Closes the conversation politely

The tone remains:

✔ calm
✔ empathetic
✔ supportive

---

## 🧠 System Pipeline

```
Microphone Input
 → Voice Activity Detection
 → Streaming Speech-to-Text
 → LLM Conversation Logic
 → Text-to-Speech
 → Audio Output
```

### Additional Behavior (Built-In)

✔ **Language Identification**
✔ **Speaker Diarization**

> These are handled **natively by the streaming STT provider**,
> ensuring only English input is accepted and the primary patient speaker is detected.

### Emotion Awareness

The system additionally includes **emotion recognition** to support empathy tuning:

```
onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX
```

(All processing runs on **CPU**)

---

## 🧩 Technology Stack

| Component         | Tool                   |
|-------------------|------------------------|
| Realtime Engine   | LiveKit Agents         |
| Streaming ASR     | Deepgram via LiveKit   |
| LLM & TTS         | OpenAI via LiveKit plugin |
| Emotion Detection | ONNX Runtime           |
| Execution         | CPU only               |

---

## 📁 Project Structure

```
code/
 ├── main.py              # Core agent worker
 ├── voice_agent.py       # Conversation logic
 ├── emotion_model.py     # Emotion recognition
 ├── constants.py         # Config / environment
 └── ...
requirements.txt
build.sh
.env   (user-provided)
```

---

# ▶️ Running the Agent

The project is designed to run with **one script**.

### 1️⃣ Create a `.env` file in the project root

Add your keys:

```
OPENAI_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
```

These are required for:

✔ streaming transcription
✔ conversation response generation
✔ TTS playback

---

### 2️⃣ Make the script executable

```bash
chmod +x build.sh
```

### 3️⃣ Run the agent

```bash
./build.sh
```

The script will:

✔ install dependencies
✔ download required models
✔ launch the realtime agent console

No GPU is required — everything runs on CPU.

---

## 📹 Demo Video

A demo video is included showing:

🎙 real-time patient interaction
🧠 natural conversation flow
❤️ emotion-aware agent behavior
🏎 low latency performance

---

## 🏎 Latency Optimization

The system uses:

* streaming ASR
* non-blocking pipeline
* parallel emotion inference
* CPU-optimized ONNX runtime
* VAD to skip silence

Typical expected performance:

| Stage              | Latency   |
| ------------------ | --------- |
| Speech recognition | 100–300ms |
| LLM response       | < 1s      |
| TTS start          | < 300ms   |

---

## 🧪 Signal Processing Rationale

| Module               | Source              | Purpose                         |
| -------------------- | ------------------- | ------------------------------- |
| Language Detection   | STT Provider        | Ensures English-only pipeline   |
| Speaker Diarization  | STT Provider        | Handles multi-speaker scenarios |
| Emotion Detection    | ONNX Model          | Supports empathetic tone        |
| VAD & Noise Handling | LiveKit Audio Stack | Improves accuracy & latency     |

> No custom diarization or language-ID model is required —
> **these are natively provided by the STT engine.**

---

## 🔒 Safety Notes

✔ No medical diagnosis is provided
✔ Neutral & safe language
✔ No PHI stored

---

## ⚠️ Disclaimer

This project is a **technical demonstration only**
It is **not a certified clinical product**
It must **not be used for medical decision-making**

---

## 🙏 Acknowledgements

* LiveKit Realtime AI
* OpenAI Realtime Models
* Deepgram Streaming ASR
* ONNX Community Speech Emotion Model

---
