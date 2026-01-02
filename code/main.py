from livekit.plugins import openai, silero, deepgram, noise_cancellation
from livekit.agents import (
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
    JobProcess
)
from voice_agent import VoiceAgent
import os
from constants import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
from dotenv import load_dotenv
from livekit.agents.voice import room_io

load_dotenv()

os.environ['LIVEKIT_API_KEY'] = LIVEKIT_API_KEY
os.environ['LIVEKIT_URL'] = LIVEKIT_URL
os.environ['LIVEKIT_API_SECRET'] = LIVEKIT_API_SECRET


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    await ctx.wait_for_participant()


    session = AgentSession(
        turn_detection="vad",
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
            enable_diarization=True,
        ),
        llm=openai.LLM(
            model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts",voice="alloy"),
        preemptive_generation=True
    )
    await session.start(
        agent=VoiceAgent(ctx=ctx),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC()))
    )
    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.3),
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.1),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.1),
        ],
    )
    await background_audio.start(room=ctx.room, agent_session=session)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    # Start background SQS scheduler
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,
                              initialize_process_timeout=90,load_threshold=0.9))
