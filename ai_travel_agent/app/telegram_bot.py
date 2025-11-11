import asyncio
import uuid
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.core.settings import get_settings
from langchain_core.messages import ToolMessage

settings = get_settings()

TELEGRAM_TOKEN = settings.TELEGRAM_TOKEN

PENDING = {}  # { chat_id: {"awaiting": bool, "tool_call_id": str, "config": dict} }

# --------------------------------------------------------------------
# Debugging: print events to terminal
# --------------------------------------------------------------------

def _print_events_to_terminal(events_list):
    for event in events_list:
        print("\n=== EVENT ===")
        for msg in event.get("messages", []):
            print(f"[{msg.type.upper()}] id={getattr(msg,'id',None)}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print("  tool_calls:")
                for tc in msg.tool_calls:
                    print(f"    - name: {tc.get('name')}")
                    print(f"      id:   {tc.get('id')}")
                    print(f"      args: {tc.get('args')}")
            if msg.type == "tool":
                print("  tool_name:", getattr(msg, "name", None))
            print("  content:", msg.content)

# --------------------------------------------------------------------
# /start command
# --------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi 👋 I’m your Swiss Airlines assistant. Ask me about flights, hotels, car rentals, or activities!"
    )

# --------------------------------------------------------------------
# Handle normal user messages
# --------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    user_input = (update.message.text or "").strip()

    # Approval path
    pending = PENDING.get(chat_id)
    if pending and pending.get("awaiting"):
        tool_call_id = pending["tool_call_id"]
        cfg = pending["config"]

        if user_input.lower() == "y":
            result = multi_agentic_graph.invoke(None, cfg)
        else:
            denial = ToolMessage(
                tool_call_id=tool_call_id,
                content=(
                    f"API call denied by user. Reason: '{user_input}'. "
                    "Do not execute the tool. Propose safe alternatives or ask clarifying questions."
                ),
            )
            result = multi_agentic_graph.invoke({"messages": [denial]}, cfg)

        # Print and reply
        events_list = [{"messages": result.get("messages", [])}]
        _print_events_to_terminal(events_list)

        reply = None
        for m in result.get("messages", []):
            if getattr(m, "type", "") == "ai":
                reply = m.content

        await update.message.reply_text(reply)
        PENDING[chat_id].update({"awaiting": False, "tool_call_id": None})
        return

    # Normal path
    cfg = {
        "configurable": {
            "passenger_id": "7892 191894",
            "thread_id": chat_id,
        }
    }

    # IMPORTANT: materialize the stream so we can reuse it
    events_iter = multi_agentic_graph.stream(
        {"messages": [("user", user_input)]},
        cfg,
        stream_mode="values",
    )
    events_list = list(events_iter)

    # Print to terminal
    _print_events_to_terminal(events_list)

    # Extract last AI reply from the same materialized events
    reply = None
    for event in events_list:
        for msg in event.get("messages", []):
            if getattr(msg, "type", "") == "ai":
                reply = msg.content

    # Approval gate (interrupts)
    snapshot = multi_agentic_graph.get_state(cfg)
    if snapshot.next:
        last_msg = snapshot.values["messages"][-1]
        pending_calls = getattr(last_msg, "tool_calls", []) or []
        if pending_calls:
            tool_call_id = pending_calls[0]["id"]
            PENDING[chat_id] = {"awaiting": True, "tool_call_id": tool_call_id, "config": cfg}
            await update.message.reply_text(
                "Do you approve of the above actions? Type 'y' to continue; "
                "otherwise, explain your requested changes."
            )
            return

    if not reply or not reply.strip():
        reply = "Sorry, I wasn’t able to find that information right now. Could you rephrase?"

    await update.message.reply_text(reply.strip())


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        # Give more time to connect/read/write (avoid TimedOut)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
