from typing import AsyncIterable, Any, AsyncGenerator
from livekit import rtc
from livekit.agents import Agent, JobContext, stt, ModelSettings, llm, FunctionTool
from agent_tasks import AskAvailability, AskFeeling, AssessPainScore, ExerciseGuidanceTask
import numpy as np
from livekit.agents.stt import SpeechEvent
from emotion_node import EmotionNode


class VoiceAgent(Agent):
    def __init__(self, ctx:JobContext):
        self.agent_instructions = """You are a Medical Agent named Glia. 
        You should be empethetic and always respond in english 
        even if patient speaks in different language"""
        super().__init__(instructions=self.agent_instructions)
        self.ctx = ctx
        self.emotion_stt = EmotionNode("stt",self.ctx)
        self.emotion_tts = EmotionNode("tts",self.ctx)


    async def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncGenerator[SpeechEvent, Any]:
        async def filtered_audio():
            async for frame in audio:
                data = np.frombuffer(frame.data, dtype=np.int16)
                samples = data.astype("float32") / 32768.0
                self.emotion_stt.process(samples)
                yield frame

        async for event in Agent.default.stt_node(self, filtered_audio(), model_settings):

            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT or event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                speech_data: stt.SpeechData | None = event.alternatives[0]
                if speech_data:
                    detected_language = speech_data.language
                    self.ctx.proc.userdata["language"] = detected_language


            yield event

    async def llm_node(
            self,
            chat_ctx: llm.ChatContext,
            tools: list[FunctionTool],
            model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:

        language = self.ctx.proc.userdata.get("language","en")
        chat_ctx = chat_ctx.copy()
        if language != "en":
            chat_ctx.add_message(
                role="assistant",
                content=f"""
                Current Language Code Spoken by User - {language}
                
                Respectfully mention to user that you only support english language extensively.
                """
            )

        user_emotion = self.ctx.proc.userdata.get('user_emotion', ['NEUTRAL'])
        agent_emotion = self.ctx.proc.userdata.get('agent_emotion', ['NEUTRAL'])
        chat_ctx.add_message(
            role="assistant",
            content=(
                "Context update: the user's emotional tone has been estimated as "
                f"{user_emotion if user_emotion != [] else ['NEUTRAL']}"". "
                f"and the agent's emotion has been estimated as "
                f"{agent_emotion if agent_emotion != [] else ['NEUTRAL']}."
                "This is informational only and may help guide an empathetic response."
            )
        )

        await self.update_chat_ctx(chat_ctx)
        self.ctx.proc.userdata["user_emotion"]=[]
        self.ctx.proc.userdata["agent_emotion"]=[]

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            # Insert custom postprocessing here
            yield chunk

    async def tts_node(
            self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # Insert custom text processing here
        async for frame in Agent.default.tts_node(self, text, model_settings):
            # Insert custom audio processing here
            data = np.frombuffer(frame.data, dtype=np.int16)
            samples = data.astype("float32") / 32768.0
            self.emotion_tts.process(samples)
            yield frame


    async def on_enter(self) -> None:
        confirmation = await AskAvailability(self)
        if confirmation:
            feeling = await AskFeeling(self)
            pain_score = await AssessPainScore(self)
            exercise_guidance = await ExerciseGuidanceTask(self)

            ## Call Closure Logic
            await self.session.generate_reply(instructions="Provide a closing statement in english and inform them you are going to close this call",
                                              allow_interruptions=False)

            ## Commented as the console code doesn't support room closure
            # api_client = api.LiveKitAPI()
            # await api_client.room.delete_room(api.DeleteRoomRequest(
            #     room=self.ctx.job.room.name,
            # ))


