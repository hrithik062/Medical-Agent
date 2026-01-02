from livekit import rtc
from livekit.agents import AgentTask, function_tool, ModelSettings, stt
from dataclasses import dataclass
from typing import Optional, Literal, AsyncIterable, Coroutine, Any
from livekit.agents import Agent


class AskAvailability(AgentTask[bool]):
    def __init__(self, chat_ctx=None, agent_instructions=None, agent=None):
        super().__init__(
            instructions=agent_instructions+"""
            Pitch Introduction for who you are and also tell why you called the user using the below information - 
            
            Patient had Ankle surgery after a fall from a horse while playing Polo and is enrolled into a
            home program for recovery.
            
            Get confirmation for user availability before moving further
            """,
            chat_ctx=chat_ctx,
        )
        self.agent = agent

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        return self.agent.stt_node(audio, model_settings)

    async def llm_node(self,
        chat_ctx,
        tools,
        model_settings: ModelSettings):

        return self.agent.llm_node(chat_ctx, tools, model_settings)

    @function_tool
    async def user_available(self):
        """Use this if user provides confirmation for further information"""
        return self.complete(True)

    @function_tool
    async def user_not_available(self):
        """Use this if user is not interested or available for further information"""
        return self.complete(False)

    async def on_enter(self) -> None:
        print("Reached Availability Task")
        await self.session.generate_reply(instructions="Introduce yourself about why you called the patient, Mentioning your name")


class AskFeeling(AgentTask[str]):
    def __init__(self, chat_ctx=None, agent_instructions=None,agent=None):
        super().__init__(
            instructions=agent_instructions+"""
                    Ask about how the patient is feeling after surgery
                    """,
            chat_ctx=chat_ctx,
        )
        self.agent = agent


    async def llm_node(self,
        chat_ctx,
        tools,
        model_settings: ModelSettings):

        return self.agent.llm_node(chat_ctx, tools, model_settings)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        return self.agent.stt_node(audio, model_settings)

    @function_tool
    async def complete_query(self, output:str):
        """Use this once user shared the feelings"""
        return self.complete(output)

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask about how the patient is feeling after surgery")

class AssessPainScore(AgentTask[int]):
    def __init__(self, chat_ctx=None, agent_instructions=None,agent=None):
        super().__init__(
            instructions=agent_instructions+"""
            You must ask the patient to rate their pain on a scale from 1 to 10, where:
            1 = very mild pain
            10 = the worst pain imaginable.
            
            Ask the question clearly and politely:
            “On a scale from 1 to 10, where 1 means very mild pain and 10 means the worst pain you can imagine, what is your pain level right now?”
            
            Wait for the patient’s response.
            
            When the patient gives a number:
            - Acknowledge it empathetically
            - Briefly reflect the meaning of the score
            - Complete the Query
            
            Interpretation guidelines:
            1–3  = mild pain → reassure calmly
            4–6  = moderate pain → acknowledge discomfort
            7–10 = severe pain → express concern and suggest monitoring or contacting a clinician if needed (without giving clinical advice)
            
            
            Stay supportive, respectful, and calm at all times. Once user shared the pain score move to next task
            """,
            chat_ctx=chat_ctx
        )
        self.agent = agent

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        return self.agent.stt_node(audio, model_settings)

    async def llm_node(self,
        chat_ctx,
        tools,
        model_settings: ModelSettings):

        return self.agent.llm_node(chat_ctx, tools, model_settings)

    @function_tool
    async def complete_query(self, pain_score:int):
        """Use this once user shared the score"""
        return self.complete(pain_score)

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Ask patient to rate his pain score")



@dataclass
class ExerciseResult:
    exercise_name: str
    explained: bool
    patient_response: Optional[str] = None
    practiced_now: Optional[bool] = None


@dataclass
class ExerciseSessionResults:
    ankle_mobility: ExerciseResult
    toe_tapping: ExerciseResult
    calf_raises: ExerciseResult


class ExerciseGuidanceTask(AgentTask[ExerciseSessionResults]):
    def __init__(self, chat_ctx=None, agent_instructions=None, agent=None):
        super().__init__(
            instructions=agent_instructions+(
                "Guide the patient through three ankle-recovery exercises, "
                "one at a time, using a calm and supportive tone. "
                "After explaining each exercise, ask: "
                "'Would you like to take a moment to go through the steps for "
                "[exercise name] now? I can pause the call for up to one minute.' "
                "Wait for their response before moving to the next exercise."
            ),
            chat_ctx=chat_ctx
        )
        self.agent = agent

        self._results = {}

        self._exercise_order = [
            "Ankle Mobility Stretch",
            "Toe Tapping",
            "Calf Raises"
        ]

        self._exercise_prompts = {
            "Ankle Mobility Stretch": (
                "Let's start with an Ankle Mobility Stretch. "
                "Sit comfortably with your foot slightly raised. "
                "Slowly move your ankle in circles — five times clockwise "
                "and five times the other way. "
                "Move gently, and stop if you feel sharp pain."
            ),
            "Toe Tapping": (
                "Next is Toe Tapping. "
                "Keep your heel on the ground and lift your toes up and down "
                "slowly, around ten times. "
                "This helps with movement and circulation."
            ),
            "Calf Raises": (
                "Finally, we'll do Calf Raises. "
                "Stand while holding onto a stable surface. "
                "Slowly rise up onto your toes, then lower back down, "
                "about ten times if comfortable. "
                "Move at your own pace and within your comfort."
            )
        }

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        return self.agent.stt_node(audio, model_settings)

    async def llm_node(self,
        chat_ctx,
        tools,
        model_settings: ModelSettings):

        return self.agent.llm_node(chat_ctx, tools, model_settings)

    async def on_enter(self) -> None:
        """
        Start the task by gently transitioning to exercise guidance.
        """
        await self.session.generate_reply(
            instructions=(
                "Explain that you will now guide them through a few simple "
                "ankle-recovery exercises to support their healing. "
                "Stay warm, calm, and encouraging."
            )
        )
        self._continue_flow()

    #
    # ---------- RECORDING FUNCTIONS ----------
    #

    @function_tool
    async def record_exercise_response(
        self,
        exercise_name: Literal['Ankle Mobility Stretch', 'Toe Tapping', 'Calf Raises'],
        patient_response: str,
        practiced_now: Optional[bool] = None
    ):
        """
        Record the patient’s response after explaining an exercise.
        """

        self._results[exercise_name] = ExerciseResult(
            exercise_name=exercise_name,
            explained=True,
            patient_response=patient_response,
            practiced_now=practiced_now
        )

        self._check_completion()

    #
    # ---------- FLOW CONTROL ----------
    #

    def _continue_flow(self):
        # Determine which exercise is next
        for exercise in self._exercise_order:
            if exercise not in self._results:
                self._explain_exercise(exercise)
                return

        # If all are recorded, finish
        self._complete_task()

    def _explain_exercise(self, exercise_name: str):
        explanation = self._exercise_prompts[exercise_name]

        self.session.generate_reply(
            instructions=(
                f"{explanation} "
                f"Would you like to take a moment to go through the steps for "
                f"{exercise_name} now? I can pause the call for up to one minute."
            )
        )

    def _check_completion(self):
        if len(self._results) == len(self._exercise_order):
            self._complete_task()
        else:
            self._continue_flow()

    def _complete_task(self):
        results = ExerciseSessionResults(
            ankle_mobility=self._results["Ankle Mobility Stretch"],
            toe_tapping=self._results["Toe Tapping"],
            calf_raises=self._results["Calf Raises"],
        )

        self.complete(results)
