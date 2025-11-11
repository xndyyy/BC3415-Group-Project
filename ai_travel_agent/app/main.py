# main.py

import uuid
import os  # Import os module for file operations
from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.services.utils import download_and_prepare_db
from customer_support_chat.app.core.logger import logger
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.errors import GraphRecursionError

def main():
    # Ensure the database is downloaded and prepared
    # download_and_prepare_db()

    # Generate a unique thread ID for the session
    thread_id = str(uuid.uuid4())

    # Configuration with passenger_id and thread_id
    config = {
        "configurable": {
            "passenger_id": "7892 191894",  # Update with a valid passenger ID as needed
            "thread_id": thread_id,
        }
    }

    # Variable to track printed message IDs to avoid duplicates
    printed_message_ids = set()

    try:
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Process the user input through the graph
            events = multi_agentic_graph.stream(
                {"messages": [("user", user_input)]}, config, stream_mode="values"
            )

            for event in events:
                messages = event.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id)

            for event in events:
                messages = event.get("messages", [])
                for message in messages:
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.get("name", "")
                            if tool_name == "CompleteOrEscalate":
                                tool_call_id = tool_call["id"]
                                _ = multi_agentic_graph.invoke(
                                    {
                                        "messages": [
                                            ToolMessage(
                                                tool_call_id=tool_call_id,
                                                content="CompleteOrEscalate acknowledged. Returning control to primary assistant.",
                                            )
                                        ]
                                    },
                                    config,
                                )

            # Check for interrupts
            snapshot = multi_agentic_graph.get_state(config)
            while snapshot.next:
                # Interrupt occurred before sensitive tool execution
                user_input = input(
                    "\nDo you approve of the above actions? Type 'y' to continue; otherwise, explain your requested changes.\n\n"
                )
                if user_input.strip().lower() == "y":
                    # Continue execution
                    result = multi_agentic_graph.invoke(None, config)
                else:
                    # Provide feedback to the assistant
                    tool_call_id = snapshot.values["messages"][-1].tool_calls[0]["id"]
                    result = multi_agentic_graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=tool_call_id,
                                    content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                )
                            ]
                        },
                        config,
                    )
                # Process the result to display any new messages
                messages = result.get("messages", [])
                for message in messages:
                    if message.id not in printed_message_ids:
                        message.pretty_print()
                        printed_message_ids.add(message.id) 
                        
                # Update the snapshot
                snapshot = multi_agentic_graph.get_state(config)
                
    except GraphRecursionError:
        print("I'm sorry — I got stuck trying to complete your request. Let’s try again.")
        logger.warning("Graph recursion limit reached; recovered gracefully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print("An unexpected error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    main()
