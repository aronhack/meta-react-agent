"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import sys
from tokenizers import Tokenizer
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Optional

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.prompts import SYSTEM_PROMPT
from react_agent.utils import load_chat_model

# Define the function that calls the model


class MetaReActAgent:
    '''
    A Reasoning and Action agent that can use tools to accomplish tasks.
    '''

    def __init__(self, tools: List[str],
                 model_name=None, system_prompt=None):
        """Initialize the ReAct agent.

        Args:
            tools: List of tool functions the agent can use
            model_name: Name of the chat model to use
        """
        self.tools = self._get_tool_list(tools)
        self.model_name = model_name
        self.system_prompt = system_prompt if system_prompt \
            else SYSTEM_PROMPT
        self.chat_history_raw = []
        self.chat_history = []

        self._build_graph()
        self.tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    def _build_graph(self):
        """Build the agent's execution graph."""
        # Define a new graph
        self.builder = StateGraph(
            State, input=InputState, config_schema=Configuration)

        # Define the two nodes we will cycle between
        self.builder.add_node("call_model", self._call_model)
        self.builder.add_node("tools", ToolNode(self.tools))

        # Set the entrypoint
        self.builder.add_edge("__start__", "call_model")

        # Add conditional edges
        self.builder.add_conditional_edges(
            "call_model",
            self._route_model_output,
        )

        # Add edge from tools back to model
        self.builder.add_edge("tools", "call_model")

        # Compile the graph
        self.graph = self.builder.compile()
        self.graph.name = "ReAct Agent"

    def _call_model(self, state: State, config: RunnableConfig
                    ) -> Dict[str, List[AIMessage]]:
        '''
        Call the LLM powering our agent.
        '''
        configuration = Configuration.from_context()
        if self.model_name:
            configuration.model = self.model_name

        # Initialize the model with tool binding
        model = load_chat_model(configuration.model).bind_tools(self.tools)

        # chat_history may append SystemMessage several times
        sys_msg = SystemMessage(content=self.system_prompt)
        if sys_msg not in self.chat_history_raw:
            self.chat_history_raw.append(sys_msg)
        if sys_msg not in self.chat_history:
            self.chat_history.append(sys_msg)

        response = cast(
            AIMessage,
            model.invoke(
                [{"role": "system", "content": self.system_prompt},
                    *state.messages], config
            ),
        )

        # Handle last step case
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }

        return {"messages": [response]}

    def _get_tool_list(self, tools: list[str]):
        '''
        Get the list of tools
        '''
        result = []
        for tool in tools:
            current_module = sys.modules[__name__]
            new_func = getattr(current_module, tool)
            result.append(new_func)
        return result

    def _route_model_output(self, state: State) -> Literal["__end__", "tools"]:
        """Determine the next node based on the model's output."""
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
            )
        return "__end__" if not last_message.tool_calls else "tools"

    def process_message(self, messages: List[dict], config: Optional[dict] = None):
        """Process messages through the graph.

        Args:
            messages: List of message dictionaries
            config: Optional configuration dictionary

        Yields:
            The last message from each chunk
        """
        for chunk in self.graph.stream({"messages": messages}, config or {}, stream_mode="values"):
            yield chunk["messages"][-1]

    def run(self, messages: List[dict], config: Optional[dict] = None,
            token_limit=50000) -> List[dict]:
        '''
        Run the agent on a list of messages.

        Args:
            messages: List of message dictionaries
            config: Optional configuration dictionary

        Returns:
            List of all messages from the conversation
        '''
        system_prompt_encoded = self.tokenizer.encode(self.system_prompt)
        assert len(
            system_prompt_encoded.tokens) < token_limit, "System prompt exceed token limit"

        messages_encoded = self.tokenizer.encode(messages[-1]['content'])
        assert len(
            messages_encoded.tokens) < token_limit, "Messages exceed token limit"

        # Generate the response and add to chat history
        all_messages = []
        for message in self.process_message(messages, config):
            all_messages.append(message)

        self.chat_history_raw += all_messages
        self.chat_history.append(HumanMessage(content=messages[-1]['content']))
        self.chat_history.append(all_messages[-1])

        return all_messages


def examples():

    # Research agent
    img_gen_agent = MetaReActAgent(
        tools=["search", "query_database"])

    # Image generation agent
    img_gen_agent = MetaReActAgent(
        tools=["img_generate_image"])

    # Monitor and alert agent
    monitor_agent = MetaReActAgent(
        tools=["query_database", "send_telegram_message"])
