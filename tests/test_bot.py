import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from telegram import Update
from telegram.ext import ContextTypes

from app.bot import (
    start_command,
    help_command,
    new_post_command,
    USER_STATES,
    generate_single_image_and_send,
    generate_images_session,
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
    assert (
        "Please provide an Instagram post URL"
        in update.message.reply_text.call_args[0][0]
    )
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


@pytest.mark.anyio
@patch("app.bot.generate_reposed_image")
async def test_generate_single_image_and_send(mock_generate_reposed):
    # Mock return value of generation
    mock_generate_reposed.return_value = {
        "status": "success",
        "generated_images": ["/tmp/result.png"],
    }

    # Mock Update and Context
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.bot = AsyncMock()

    cfg = {"is_selfie": True, "is_mirror_selfie": False}

    # Open mock for local file reading during send
    from unittest.mock import mock_open
    with patch("builtins.open", mock_open(read_data=b"imagebytes")):
        await generate_single_image_and_send(
            chat_id=12345,
            orig_idx=0,
            img_path="/tmp/scene.jpg",
            cfg=cfg,
            identity_name="gal",
            context=context,
        )

    # Verify that the bot sends status message and photo
    assert context.bot.send_message.call_count == 1
    assert "Generating variation" in context.bot.send_message.call_args[1]["text"]
    assert context.bot.send_photo.call_count == 1
    assert context.bot.send_photo.call_args[1]["chat_id"] == 12345
    assert "New image for Image 1" in context.bot.send_photo.call_args[1]["caption"]


@pytest.mark.anyio
@patch("app.bot.generate_single_image_and_send")
async def test_generate_images_session_concurrent(mock_gen_single):
    chat_id = 12345
    USER_STATES[chat_id] = {
        "url": "https://instagram.com/p/test",
        "identity_name": "gal",
        "media_files": ["/tmp/1.jpg", "/tmp/2.jpg"],
        "selected_indices": {0, 1},
        "gen_queue": [(0, "/tmp/1.jpg"), (1, "/tmp/2.jpg")],
        "gen_configs": {
            0: {"is_selfie": False, "is_mirror_selfie": False},
            1: {"is_selfie": True, "is_mirror_selfie": True},
        },
    }

    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.bot = AsyncMock()

    await generate_images_session(chat_id, context)

    # Verify that generate_single_image_and_send is called for both images concurrently
    assert mock_gen_single.call_count == 2
    mock_gen_single.assert_any_call(
        chat_id=chat_id,
        orig_idx=0,
        img_path="/tmp/1.jpg",
        cfg={"is_selfie": False, "is_mirror_selfie": False},
        identity_name="gal",
        context=context,
    )
    mock_gen_single.assert_any_call(
        chat_id=chat_id,
        orig_idx=1,
        img_path="/tmp/2.jpg",
        cfg={"is_selfie": True, "is_mirror_selfie": True},
        identity_name="gal",
        context=context,
    )

    # Verify that completion message is sent
    assert context.bot.send_message.call_count == 2
    assert "Starting generation session" in context.bot.send_message.call_args_list[0][1]["text"]
    assert "All variations have been generated" in context.bot.send_message.call_args_list[1][1]["text"]

