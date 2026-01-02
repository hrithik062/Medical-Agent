import threading
import queue
import numpy as np
import onnxruntime as ort
from typing import Literal
from scipy.special import softmax
from collections import deque
from typing import Optional, Tuple
from livekit.agents import JobContext

SAMPLE_RATE = 16000
WIN = SAMPLE_RATE * 2         # 1 sec window
HOP = SAMPLE_RATE // 2        # 0.5 sec hop

# Labels for DunnBC22 SER models
id2label = {
    0: 'NEUTRAL',
    1: 'ANGRY',
    2: 'SAD',
    3: 'FEAR',
    4: 'HAPPY',
    5: 'DISGUST'}


class EmotionNode:
    """
    LiveKit-compatible streaming emotion node.
    - Returns input audio frames untouched
    - Performs ONNX SER in background
    - Emits timestamped emotion state
    """

    def __init__(self, node_type:Literal["stt","tts"], ctx: JobContext = None):
        self.session = ort.InferenceSession("code/emotion_model.onnx")

        self.stream_time = 0.0     # seconds of audio processed
        self.frame_queue = queue.Queue(maxsize=200)

        self.audio_buffer = np.zeros(0, dtype=np.float32)

        self.latest: Optional[Tuple[float, str, float]] = None
        self.smooth = deque(maxlen=5)
        self.node_type = node_type
        self.ctx = ctx
        self._stop = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _infer(self, wav: np.ndarray):
        wav = wav.astype("float32")[None, :]
        logits = self.session.run(None, {"input_values": wav})[0][0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))

        return id2label[idx], float(probs[idx])

    def _run(self):
        while not self._stop:
            frame, start_ts = self.frame_queue.get()

            # grow buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, frame])

            # keep 2 sec max
            if len(self.audio_buffer) > SAMPLE_RATE * 2:
                self.audio_buffer = self.audio_buffer[-SAMPLE_RATE * 2:]

            # process windows
            while len(self.audio_buffer) >= WIN:
                window = self.audio_buffer[:WIN]
                self.audio_buffer = self.audio_buffer[HOP:]

                # center timestamp of window
                ts = start_ts + (WIN / SAMPLE_RATE) / 2.0

                label, conf = self._infer(window)
                if self.node_type == "stt":
                    emotion = self.ctx.proc.userdata.get("user_emotion",[])
                    if conf>0.7:
                        emotion.append(label)
                    self.ctx.proc.userdata["user_emotion"]= emotion
                else:
                    emotion = self.ctx.proc.userdata.get("agent_emotion", [])
                    if conf > 0.70:
                        emotion.append(label)
                    self.ctx.proc.userdata["agent_emotion"] = emotion
                self.smooth.append(label)
                stable = max(set(self.smooth), key=self.smooth.count)

                self.latest = (ts, stable, conf)

    def stop(self):
        self._stop = True
        self.worker.join()

    def process(self, frame: np.ndarray):
        """
        MAIN ENTRYPOINT

        frame: float32 mono PCM (e.g., 800 samples @ 16kHz)

        returns: same frame unchanged
        """

        duration = len(frame) / SAMPLE_RATE
        frame_start = self.stream_time
        self.stream_time += duration

        try:
            self.frame_queue.put_nowait((frame.copy(), frame_start))
        except queue.Full:
            pass  # drop if overloaded to preserve latency

        return frame

    def get_latest(self):
        """
        Returns (timestamp_sec, emotion_label, confidence) or None
        """
        return self.latest
