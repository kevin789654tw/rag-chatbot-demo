import chainlit as cl

from app.core.app import handle_user_message, init_conversation


@cl.on_chat_start
async def start() -> None:
    """Initialize the chat session.

    Returns:
        None
    """
    await init_conversation()


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handle a user message from the Chainlit UI. (trigger by user message)

    Args:
        message (cl.Message): The message object received from the Chainlit UI
        containing the user's input.

    Returns:
        None
    """
    await handle_user_message(message)
