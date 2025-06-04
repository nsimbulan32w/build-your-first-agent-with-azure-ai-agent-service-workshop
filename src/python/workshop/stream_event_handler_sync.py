from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentEventHandler,
    FunctionTool,
    MessageDeltaChunk,
    MessageStatus,
    RunStatus,
    RunStep,
    RunStepDeltaChunk,
    RunStepStatus,
    ThreadMessage,
    ThreadRun,
)

from utilities import Utilities
from terminal_colors import TerminalColors as tc

response_stream = []

class StreamEventHandler(AgentEventHandler[str]):
    """Handle LLM streaming events and tokens."""

    def __init__(self, functions: FunctionTool, project_client: AIProjectClient, utilities: Utilities) -> None:
        self.functions = functions
        self.project_client = project_client
        self.util = utilities
        response_stream.clear()
        
        super().__init__()

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        """Handle message delta events. This will be the streamed token"""
        self.util.log_token_blue(delta.text)
        # resp = f"{tc.GREEN}{delta.text}{tc.RESET}"
        resp = delta.text

        response_stream.append(resp)


    def on_thread_message(self, message: ThreadMessage) -> None:
        """Handle thread message events."""
        pass
        # if message.status == MessageStatus.COMPLETED:
        #     print()
        # self.util.log_msg_purple(f"ThreadMessage created. ID: {message.id}, " f"Status: {message.status}")

        self.util.get_files(message, self.project_client)

    def on_thread_run(self, run: ThreadRun) -> None:
        """Handle thread run events"""

        if run.status == RunStatus.FAILED:
            print(f"Run failed. Error: {run.last_error}")
            print(f"Thread ID: {run.thread_id}")
            print(f"Run ID: {run.id}")

    def on_run_step(self, step: RunStep) -> None:
        pass
        # if step.status == RunStepStatus.COMPLETED:
        #     print()
        # self.util.log_msg_purple(f"RunStep type: {step.type}, Status: {step.status}")

    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        pass

    def on_error(self, data: str) -> None:
        print(f"An error occurred. Data: {data}")

    def on_done(self) -> None:
        """Handle stream completion."""
        # return self.response_stream
        pass
        # self.util.log_msg_purple(f"\nStream completed.")

    def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
        """Handle unhandled events."""
        # print(f"Unhandled Event Type: {event_type}, Data: {event_data}")
        print(f"Unhandled Event Type: {event_type}")
