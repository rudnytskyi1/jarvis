# Telegram group

Rowan can send a requested message or an existing/generated image to one group
configured on the brain PC. The model cannot supply another recipient. Optional
mention/reply answers use that same group and one shared Telegram history.
An optional numeric `control_user_id` grants one account the existing Rowan
tools, in that group or in its own private conversation with the bot. Other
group members retain ordinary chat/image requests; other private senders are
ignored. Display names, mentions and forwarded sender names grant no access.

## Configuration

Run `set-telegram-key.bat` on the brain PC (it calls
`scripts/set-telegram-key.ps1`) and paste the bot token into its hidden prompt.
The token is encrypted for the current Windows account in
`%LOCALAPPDATA%\Jarvis\telegram-bot-token.dpapi`. The normal
`start-jarvis-openai.bat` launcher loads it into `TELEGRAM_BOT_TOKEN` for the brain
process; neither the room client nor a public client distribution needs it.
Never put the token in YAML, prompts, logs or the client package.

```yaml
server:
  telegram:
    enabled: true
    chat_id: -1234567890 # replace with the intended group's numeric ID
    control_user_id: null # optional positive USER ID, never a group/chat display name
    api_key_env: TELEGRAM_BOT_TOKEN
    timeout_s: 30
    respond_to_mentions: false
    poll_timeout_s: 25
```

Telegram is disabled by default and a missing token leaves ordinary voice chat
available. Add the bot to the configured group and allow it to post messages and
photos. After saving credentials or changing settings, restart the brain server.
`check_connection()` reads only bot and group information; it sends no test
message, consumes no pending updates and does not prove posting permissions.

## Messages and images

The transport uses `sendMessage`, `sendPhoto` or `sendDocument` and returns a
confirmed message ID only when Telegram acknowledges the configured group.
Text is sent literally, without Markdown/HTML parsing. Rowan rejects messages
over 4096 UTF-16 code units or captions over 1024 rather than splitting one
request into several untracked sends. [Telegram method documentation](https://core.telegram.org/bots/api#sendmessage).

Static PNG/JPEG pictures use a photo upload when they fit the photo limits:
10 MB, combined width and height at most 10,000, and aspect ratio at most 20.
Larger pictures and WebP use a document upload, up to the app's 50 MB limit, to
preserve the supplied file. The decision happens before sending; failed photo
uploads never trigger an automatic second upload. [Photo](https://core.telegram.org/bots/api#sendphoto)
and [document](https://core.telegram.org/bots/api#senddocument) methods.

Sending an existing image does not generate it again. Ordinary camera frames,
appearance archives, request audio and conversation history are not automatically
posted. Only the explicitly requested content is delivered.

## Mentions, replies and shared context

Set `respond_to_mentions: true` to enable the optional group listener. New
messages that mention this bot or reply to one of its messages can receive an
answer. Reply matching checks the parent message's numeric bot ID, including
replies to bot messages originally sent by a room voice command. A matching
display name alone does not activate Rowan. Other visible group messages can
be retained as context without causing an answer.

All participants share one durable conversation key for the configured group.
Context combines the latest 25 exchanges and up to 25 delivered background group
messages, with authors, numeric sender IDs and timestamps, subject to the input
size budget. Older requests remain in the local database. Historical rows from
the earlier per-sender layout are read together chronologically; they are not
deleted. Restarting Rowan preserves this history. The bot can retain only updates
Telegram delivers to it; this does not change its group privacy settings.

Telegram history stays separate from private room voice histories. Image caches
retain their sender provenance; a reply can explicitly reference the bot's
attached image. Private controller history uses `telegram:dm:<user-id>` and is
never included in shared group history. Tools are available only to the exact
configured controller account, even when room voice permissions are disabled.
The backend independently verifies the sender before each action. Commands use
the existing room client transport with isolated speaker/history state. Busy or
offline room tools report their actual availability; ordinary chat and attached
image generation remain available. Private image replies stay private unless
the current requester explicitly asks to send them to the configured group.

Startup acknowledges old pending updates without answering them. Claimed work is
recorded before processing, preventing automatic replay after interruption or
restart. Polling can recover after connectivity errors; message/image delivery
itself is never automatically retried. A pending webhook pauses polling and is
not deleted automatically. [Update delivery](https://core.telegram.org/bots/api#getupdates).

Image attachments referenced by an addressed request are downloaded only from
Telegram's fixed HTTPS endpoint and validated as static images. The default
download bound is 8 MB. Credential-bearing file URLs are never returned to chat
or logs. [Telegram file API](https://core.telegram.org/bots/api#getfile).

## Failure handling

The client uses a worker-based HTTPS transport without HTTP URL logging or
redirects. Errors redact credentials. Timeouts and interrupted sends can leave
delivery uncertain: the message might have reached Telegram even without an
acknowledgement, so Rowan must not claim success or automatically resend it.
Rate limits report their wait interval without sleeping and retrying the send.
If a group migrates, update its configured numeric ID; a provider response cannot
silently choose a different destination.
