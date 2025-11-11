from typing import Optional
from langchain_core.runnables import Runnable, RunnableConfig
from customer_support_chat.app.core.state import State
from pydantic import BaseModel
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

settings = get_settings()

# Initialize the language model (shared among assistants)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=settings.OPENAI_API_KEY,
    temperature=0.7,
)

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: Optional[RunnableConfig] = None):
        """
        Safely invokes the assistant.
        Handles recursion limits, token tracking, and graceful error fallback.
        """
        MAX_RECURSION_DEPTH = 25  # Prevent infinite loops

        try:
            for i in range(MAX_RECURSION_DEPTH):
                result = self.runnable.invoke(state, config)

                # --- Track token usage if available ---
                # if hasattr(result, "response_metadata"):
                #     usage = result.response_metadata.get("token_usage", {})
                #     prompt_tokens = usage.get("prompt_tokens", 0)
                #     completion_tokens = usage.get("completion_tokens", 0)
                #     total_tokens = usage.get("total_tokens", 0)
                #     print(
                #         f"Prompt tokens: {prompt_tokens}, "
                #         f"Completion tokens: {completion_tokens}, "
                #         f"Total: {total_tokens}"
                #     )

                # --- Break condition: valid final message ---
                if not result.tool_calls and (
                    not result.content
                    or (isinstance(result.content, list)
                        and not result.content[0].get("text"))
                ):
                    # Retry once if output was empty
                    messages = state["messages"] + [
                        ("user", "Respond with a real output.")
                    ]
                    state = {**state, "messages": messages}
                else:
                    return {"messages": result}

            # --- If recursion limit reached ---
            logger.warning("Recursion limit hit in Assistant; stopping loop.")
            return {
                "messages": AIMessage(
                    content="I'm sorry — I got stuck trying to complete your request. Let's start over."
                )
            }

        except Exception as e:
            logger.error(f"Assistant runtime error: {e}")
            return {
                "messages": AIMessage(
                    content="Sorry — something went wrong while processing your request. Please try again later."
                )
            }

# Define the CompleteOrEscalate tool
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed or to escalate control to the main assistant."""
    cancel: bool = True
    reason: str
    