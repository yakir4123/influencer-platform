import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.core.config import settings
from app.services.instagram import download_instagram_post
from app.services.image import generate_reposed_image
from app.schemas.image import ImageGenerationRequest

logger = logging.getLogger("app.bot")

# In-memory dictionary to track session state for each chat
# key: chat_id
# value: dict of session variables
USER_STATES = {}

# Static mapping for identity names to images
NAME_TO_IMAGE_MAP = {
    "gal": "app/assets/gal.png"
}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command."""
    welcome_text = (
        "🤖 *Welcome to the Influencer Platform Bot!*\n\n"
        "I can download Instagram posts and generate variations of them for you.\n\n"
        "To get started, use the command:\n"
        "`/new-post <Instagram Post URL>`\n"
        "or\n"
        "`/new_post <Instagram Post URL>`"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /help command."""
    help_text = (
        "📖 *Help & Instructions*\n\n"
        "Commands:\n"
        "- `/start` - Start the bot\n"
        "- `/help` - Show instructions\n"
        "- `/new-post <url>` - Start downloading a post\n\n"
        "Flow:\n"
        "1. Send `/new-post <url>`.\n"
        "2. Select the character's identity name (e.g. `gal`).\n"
        "3. Wait for the download to finish. Images will be sent to you with numbers.\n"
        "4. Use the multi-select buttons to select which images you want to generate variations of.\n"
        "5. Configure each selected image's options (`Selfie` / `Mirror Selfie`).\n"
        "6. Click generate, and I'll send you the new images!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def new_post_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /new-post and /new_post command."""
    chat_id = update.effective_chat.id
    text = update.message.text
    
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text(
            "⚠️ Please provide an Instagram post URL. Example:\n"
            "`/new-post https://www.instagram.com/p/DFhQzTqOa72/`",
            parse_mode="Markdown"
        )
        return

    url = parts[1].strip()

    # Reset/Initialize user state
    USER_STATES[chat_id] = {
        "url": url,
        "identity_name": None,
        "download_status": "downloading",
        "media_files": [],
        "selected_indices": set(),
        "gen_queue": [],
        "current_gen_idx": 0,
        "gen_configs": {},
    }

    # 1. Start download task in background
    asyncio.create_task(download_post_bg(chat_id, url, context))

    # 2. Ask user to choose identity name immediately
    keyboard = []
    # Dynamic buttons based on Name to Image map
    for name in NAME_TO_IMAGE_MAP.keys():
        keyboard.append([InlineKeyboardButton(name, callback_data=f"select_name:{name}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📥 Downloading post images in the background...\n\n"
        "While waiting, please select the identity reference name:",
        reply_markup=reply_markup
    )


async def download_post_bg(chat_id: int, url: str, context: ContextTypes.DEFAULT_TYPE):
    """Background task to download Instagram post."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: download_instagram_post(url))
        
        all_files = result.get("media_files", [])
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        media_files = []
        for f_path in all_files:
            # strip "gs://" prefix just to check extension if it's GCS URI
            clean_path = f_path[5:] if f_path.startswith("gs://") else f_path
            suffix = Path(clean_path).suffix.lower()
            if suffix in image_extensions:
                media_files.append(f_path)

        if not media_files:
            raise ValueError("No images found in the downloaded post.")

        if chat_id in USER_STATES:
            state = USER_STATES[chat_id]
            state["media_files"] = media_files
            state["download_status"] = "finished"

            # If they already chose the identity name, proceed to next step
            if state["identity_name"] is not None:
                await proceed_after_download(chat_id, context)
    except Exception as e:
        logger.error(f"Download failed for chat {chat_id}: {e}", exc_info=True)
        if chat_id in USER_STATES:
            USER_STATES[chat_id]["download_status"] = "failed"
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Failed to download Instagram post: {str(e)}"
        )


async def proceed_after_download(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Sends the downloaded images and prompts the user to select images."""
    state = USER_STATES[chat_id]
    media_files = state["media_files"]

    await context.bot.send_message(
        chat_id=chat_id,
        text=f"✅ Download finished! Sending {len(media_files)} images with ascending numbers:"
    )

    # Send each image sequentially
    for idx, file_path in enumerate(media_files):
        caption = f"Image {idx + 1}"
        try:
            if file_path.startswith("gs://"):
                from app.services.gcs import download_gcs_file_bytes
                img_bytes = download_gcs_file_bytes(file_path)
                if img_bytes:
                    await context.bot.send_photo(chat_id=chat_id, photo=img_bytes, caption=caption)
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Failed to read image {idx + 1} from Cloud Storage."
                    )
            else:
                with open(file_path, "rb") as f:
                    await context.bot.send_photo(chat_id=chat_id, photo=f, caption=caption)
        except Exception as e:
            logger.error(f"Failed to send image {idx + 1} ({file_path}): {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Error sending image {idx + 1}: {e}")

    # Prompt user with multi-select buttons
    await send_multiselect_keyboard(chat_id, context)


async def send_multiselect_keyboard(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Sends the inline keyboard for choosing images."""
    state = USER_STATES[chat_id]
    media_files = state["media_files"]
    selected = state["selected_indices"]

    keyboard = []
    current_row = []
    
    for i in range(len(media_files)):
        num = i + 1
        is_sel = i in selected
        tick = "✓" if is_sel else " "
        btn_text = f"[{tick}] {num}"
        current_row.append(InlineKeyboardButton(btn_text, callback_data=f"toggle_img:{i}"))
        
        # 4 buttons per row max
        if len(current_row) == 4:
            keyboard.append(current_row)
            current_row = []
            
    if current_row:
        keyboard.append(current_row)

    # Confirm selection button
    keyboard.append([InlineKeyboardButton("Confirm Selection", callback_data="confirm_images")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(
        chat_id=chat_id,
        text="👉 Select the images you want to generate variations for, then click Confirm Selection:",
        reply_markup=reply_markup
    )


async def ask_image_generation_config(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Asks configurations of is_selfie & is_mirror_selfie for the current image in the queue."""
    state = USER_STATES[chat_id]
    queue = state["gen_queue"]
    curr_idx = state["current_gen_idx"]

    if curr_idx >= len(queue):
        # Done configuring! Start generating
        await generate_images_session(chat_id, context)
        return

    orig_idx, img_path = queue[curr_idx]

    # Initialize configs if not exists
    if orig_idx not in state["gen_configs"]:
        state["gen_configs"][orig_idx] = {
            "is_selfie": False,
            "is_mirror_selfie": False
        }

    cfg = state["gen_configs"][orig_idx]
    selfie_tick = "✓" if cfg["is_selfie"] else " "
    mirror_tick = "✓" if cfg["is_mirror_selfie"] else " "

    keyboard = [
        [
            InlineKeyboardButton(f"[{selfie_tick}] Selfie", callback_data=f"toggle_cfg:selfie:{orig_idx}"),
            InlineKeyboardButton(f"[{mirror_tick}] Mirror Selfie", callback_data=f"toggle_cfg:mirror:{orig_idx}")
        ],
        [
            InlineKeyboardButton("Next / Confirm", callback_data=f"confirm_cfg:{orig_idx}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    text = (
        f"📸 *Configuration Session (Image {curr_idx + 1} of {len(queue)})*\n"
        f"Configure options for Image {orig_idx + 1}:\n\n"
        f"- *Selfie*: {'Yes' if cfg['is_selfie'] else 'No'}\n"
        f"- *Mirror Selfie*: {'Yes' if cfg['is_mirror_selfie'] else 'No'}"
    )

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )


async def generate_images_session(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Processes the queued image configurations and generates the variations."""
    state = USER_STATES[chat_id]
    queue = state["gen_queue"]
    configs = state["gen_configs"]
    identity_name = state["identity_name"]

    await context.bot.send_message(
        chat_id=chat_id,
        text=f"🚀 Starting generation session of {len(queue)} images. Please wait..."
    )

    for count_idx, (orig_idx, img_path) in enumerate(queue):
        cfg = configs[orig_idx]
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🎨 Generating variation for Image {orig_idx + 1}..."
        )

        payload = ImageGenerationRequest(
            preset="Normal",
            prompt="",
            count=1,
            retry_count=3,
            image_size="1K",
            image_2=img_path,
            is_selfie=cfg["is_selfie"],
            is_mirror_selfie=cfg["is_mirror_selfie"],
            identity_name=identity_name
        )

        try:
            res = await generate_reposed_image(payload)
            
            gen_images = res.get("generated_images", [])
            if not gen_images:
                raise ValueError("No images returned from generator.")

            for file_path in gen_images:
                caption = (
                    f"✨ New image for Image {orig_idx + 1}\n"
                    f"Identity: {identity_name}\n"
                    f"Selfie: {cfg['is_selfie']}, Mirror Selfie: {cfg['is_mirror_selfie']}"
                )
                if file_path.startswith("gs://"):
                    from app.services.gcs import download_gcs_file_bytes
                    img_bytes = download_gcs_file_bytes(file_path)
                    if img_bytes:
                        await context.bot.send_photo(chat_id=chat_id, photo=img_bytes, caption=caption)
                    else:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=f"⚠️ Failed to read generated image from storage: {file_path}"
                        )
                else:
                    with open(file_path, "rb") as f:
                        await context.bot.send_photo(chat_id=chat_id, photo=f, caption=caption)
        except Exception as e:
            logger.error(f"Generation failed for Image {orig_idx + 1}: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Failed to generate variation for Image {orig_idx + 1}: {str(e)}"
            )

    await context.bot.send_message(
        chat_id=chat_id,
        text="🎉 All variations have been generated and sent!"
    )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback query dispatcher for interactive inline keyboard buttons."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    data = query.data

    if chat_id not in USER_STATES:
        await query.message.reply_text("⚠️ Session not found or expired. Use `/new-post` to start.")
        return

    state = USER_STATES[chat_id]

    # 1. Identity Name selection
    if data.startswith("select_name:"):
        name = data.split(":", 1)[1]
        state["identity_name"] = name
        await query.edit_message_text(f"Identity selected: *{name}*", parse_mode="Markdown")

        # If download is already finished, proceed immediately
        if state["download_status"] == "finished":
            await proceed_after_download(chat_id, context)
        elif state["download_status"] == "downloading":
            await query.message.reply_text("Identity selected. Downloading images, please wait...")

    # 2. Image multi-select toggle
    elif data.startswith("toggle_img:"):
        idx = int(data.split(":", 1)[1])
        selected = state["selected_indices"]
        
        if idx in selected:
            selected.remove(idx)
        else:
            selected.add(idx)

        # Rebuild keyboard
        keyboard = []
        current_row = []
        for i in range(len(state["media_files"])):
            num = i + 1
            is_sel = i in selected
            tick = "✓" if is_sel else " "
            btn_text = f"[{tick}] {num}"
            current_row.append(InlineKeyboardButton(btn_text, callback_data=f"toggle_img:{i}"))
            
            if len(current_row) == 4:
                keyboard.append(current_row)
                current_row = []
                
        if current_row:
            keyboard.append(current_row)

        keyboard.append([InlineKeyboardButton("Confirm Selection", callback_data="confirm_images")])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_reply_markup(reply_markup=reply_markup)

    # 3. Confirm image multi-select
    elif data == "confirm_images":
        selected = state["selected_indices"]
        if not selected:
            await query.message.reply_text("⚠️ Please select at least one image first!")
            return

        # Clean/update message
        selected_text = ", ".join(str(i + 1) for i in sorted(selected))
        await query.edit_message_text(f"Images selected for variations: {selected_text}")

        # Construct generation queue
        sorted_indices = sorted(list(selected))
        state["gen_queue"] = [(idx, state["media_files"][idx]) for idx in sorted_indices]
        state["current_gen_idx"] = 0
        state["gen_configs"] = {}

        await ask_image_generation_config(chat_id, context)

    # 4. Toggle configuration during session
    elif data.startswith("toggle_cfg:"):
        # toggle_cfg:{field}:{orig_idx}
        parts = data.split(":")
        field = parts[1]
        orig_idx = int(parts[2])

        cfg = state["gen_configs"][orig_idx]
        if field == "selfie":
            cfg["is_selfie"] = not cfg["is_selfie"]
        elif field == "mirror":
            cfg["is_mirror_selfie"] = not cfg["is_mirror_selfie"]

        # Re-render configuration prompt
        selfie_tick = "✓" if cfg["is_selfie"] else " "
        mirror_tick = "✓" if cfg["is_mirror_selfie"] else " "

        keyboard = [
            [
                InlineKeyboardButton(f"[{selfie_tick}] Selfie", callback_data=f"toggle_cfg:selfie:{orig_idx}"),
                InlineKeyboardButton(f"[{mirror_tick}] Mirror Selfie", callback_data=f"toggle_cfg:mirror:{orig_idx}")
            ],
            [
                InlineKeyboardButton("Next / Confirm", callback_data=f"confirm_cfg:{orig_idx}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        curr_idx = state["current_gen_idx"]
        queue = state["gen_queue"]

        text = (
            f"📸 *Configuration Session (Image {curr_idx + 1} of {len(queue)})*\n"
            f"Configure options for Image {orig_idx + 1}:\n\n"
            f"- *Selfie*: {'Yes' if cfg['is_selfie'] else 'No'}\n"
            f"- *Mirror Selfie*: {'Yes' if cfg['is_mirror_selfie'] else 'No'}"
        )

        await query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode="Markdown")

    # 5. Confirm configuration for current image
    elif data.startswith("confirm_cfg:"):
        orig_idx = int(data.split(":")[1])
        cfg = state["gen_configs"][orig_idx]

        # Edit current message to confirm
        await query.edit_message_text(
            f"✅ Configured Image {orig_idx + 1}:\n"
            f"- Selfie: {'Yes' if cfg['is_selfie'] else 'No'}\n"
            f"- Mirror Selfie: {'Yes' if cfg['is_mirror_selfie'] else 'No'}"
        )

        # Move to next image in queue
        state["current_gen_idx"] += 1
        await ask_image_generation_config(chat_id, context)


# ── Lifespan Tasks for FastAPI integration ────────────────────────────────────

_bot_app: Optional[Application] = None


def get_bot_app() -> Optional[Application]:
    """Helper to retrieve the initialized bot application."""
    global _bot_app
    return _bot_app


async def run_bot_async():
    """Starts the Telegram bot in webhook or long polling mode asynchronously."""
    global _bot_app
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN is not configured. Telegram Bot will not start.")
        return

    logger.info("Initializing Telegram Bot...")
    _bot_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    _bot_app.add_handler(CommandHandler("start", start_command))
    _bot_app.add_handler(CommandHandler("help", help_command))
    _bot_app.add_handler(CommandHandler("new_post", new_post_command))
    _bot_app.add_handler(MessageHandler(filters.Regex(r"^/new-post"), new_post_command))
    _bot_app.add_handler(CallbackQueryHandler(handle_callback_query))

    await _bot_app.initialize()
    await _bot_app.start()

    if settings.TELEGRAM_BOT_WEBHOOK_URL:
        import hashlib
        secret_token = hashlib.sha256(settings.TELEGRAM_BOT_TOKEN.encode()).hexdigest()
        await _bot_app.bot.set_webhook(
            url=settings.TELEGRAM_BOT_WEBHOOK_URL,
            secret_token=secret_token
        )
        logger.info(f"Telegram Bot is running in webhook mode. URL: {settings.TELEGRAM_BOT_WEBHOOK_URL}")
    else:
        await _bot_app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram Bot is running in long-polling mode.")


async def stop_bot_async():
    """Shuts down the Telegram bot cleanly."""
    global _bot_app
    if _bot_app:
        logger.info("Stopping Telegram Bot...")
        try:
            if _bot_app.updater and _bot_app.updater.running:
                await _bot_app.updater.stop()
            if _bot_app.running:
                await _bot_app.stop()
            await _bot_app.shutdown()
        except Exception as e:
            logger.warning(f"Error during stop_bot_async: {e}")
        finally:
            _bot_app = None
        logger.info("Telegram Bot stopped.")


if __name__ == "__main__":
    # If run standalone: python -m app.bot
    import os
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    if not settings.TELEGRAM_BOT_TOKEN:
        print("❌ Error: TELEGRAM_BOT_TOKEN is not set.")
        sys.exit(1)

    print("🤖 Starting Telegram Bot...")
    
    # Simple loop run for CLI mode
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_bot_async())
        # Block forever
        loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        loop.run_until_complete(stop_bot_async())
    finally:
        loop.close()
