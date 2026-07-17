import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from telegram import Update
from telegram.ext import ContextTypes

from app.bot import (
    start_command,
    help_command,
    new_post_command,
    handle_callback_query,
    USER_STATES,
)


@pytest.mark.anyio
async def test_start_command():
    # Mock Update and Context
    update = MagicMock(spec=Update)
    update.message = AsyncMock()
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    
    await start_command(update, context)
    update.message.reply_text.assert_called_once()
    assert "Welcome" in update.message.reply_text.call_args[0][0]


@pytest.mark.anyio
async def test_help_command():
    # Mock Update and Context
    update = MagicMock(spec=Update)
    update.message = AsyncMock()
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    
    await help_command(update, context)
    update.message.reply_text.assert_called_once()
    assert "Help & Instructions" in update.message.reply_text.call_args[0][0]


@pytest.mark.anyio
@patch("app.bot.download_post_bg")
async def test_new_post_command_no_args(mock_download_bg):
    # Mock Update and Context with message containing no args
    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    update.message = AsyncMock()
    update.message.text = "/new-post"
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    
    await new_post_command(update, context)
    update.message.reply_text.assert_called_once()
    assert "Please provide an Instagram post URL" in update.message.reply_text.call_args[0][0]
    mock_download_bg.assert_not_called()


@pytest.mark.anyio
@patch("app.bot.asyncio.create_task")
async def test_new_post_command_with_arg(mock_create_task):
    # Mock Update and Context with message containing post URL
    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    update.message = AsyncMock()
    update.message.text = "/new-post https://instagram.com/p/test"
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    
    await new_post_command(update, context)
    
    # Check that download background task is scheduled
    mock_create_task.assert_called_once()
    
    # Check that identity name keyboard selection is displayed
    update.message.reply_text.assert_called_once()
    assert "Downloading post images" in update.message.reply_text.call_args[0][0]
    assert 12345 in USER_STATES
    assert USER_STATES[12345]["url"] == "https://instagram.com/p/test"
