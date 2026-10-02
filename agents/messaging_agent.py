import os
from datetime import datetime, timezone
from typing import List, Optional

import requests

from agents.agent import Agent
from agents.config import get
from agents.deals import Opportunity
from agents.money import format_money
from agents.normalize import STORE_NAMES

pushover_url = "https://api.pushover.net/1/messages.json"
telegram_url = "https://api.telegram.org/bot{token}/sendMessage"


def age_minutes(posted_at: Optional[datetime], now: Optional[datetime] = None) -> Optional[int]:
    if posted_at is None:
        return None
    now = now or datetime.now(timezone.utc)
    posted = posted_at if posted_at.tzinfo else posted_at.replace(tzinfo=timezone.utc)
    return max(0, int((now - posted).total_seconds() // 60))


def format_alert(opportunity: Opportunity, now: Optional[datetime] = None) -> str:
    """One deal as a short notification. Uses the deal's own currency; INR gets Indian grouping."""
    deal = opportunity.deal
    currency = opportunity.currency or deal.currency
    title = deal.title or deal.product_description[:60]
    if currency != "INR":
        # Legacy US format.
        return (
            f"Deal Alert! Price=${deal.price:.2f}, Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f} :{deal.product_description[:10]}... {deal.url}"
        )
    lines = [f"Deal: {title}"]
    price_line = f"{format_money(deal.price)}"
    if deal.mrp:
        price_line += f" (MRP {format_money(deal.mrp)})"
    price_line += f", worth about {format_money(opportunity.estimate)}"
    lines.append(price_line)
    pct = f"{opportunity.discount_pct:.0f}% / " if opportunity.discount_pct is not None else ""
    lines.append(f"{pct}{format_money(opportunity.discount)} below typical, {opportunity.confidence or 'low'} confidence")
    meta = [STORE_NAMES.get(deal.store, deal.store)]
    age = age_minutes(deal.posted_at, now)
    if age is not None:
        meta.append(f"posted {age} min ago")
    if len(deal.seen_in) > 1:
        meta.append(f"in {len(deal.seen_in)} channels")
    lines.append(", ".join(meta))
    if deal.coupon_note:
        lines.append(f"Note: {deal.coupon_note[:120]}")
    lines.append(deal.url)
    return "\n".join(lines)


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE
    MODEL = "claude-sonnet-4-5"

    def __init__(self, settings: Optional[dict] = None, dry_run: bool = False):
        """
        Set up push notifications via Pushover and, optionally, a Telegram bot.
        A target is used only if its credentials are set. With dry_run nothing is sent.
        """
        self.log("Messaging Agent is initializing")
        self.settings = settings or {}
        self.dry_run = dry_run
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.tg_bot_token = os.getenv("TG_BOT_TOKEN")
        self.tg_chat_id = os.getenv("TG_CHAT_ID")
        self.sent: List[str] = []
        targets = [name for name, ok in self.targets().items() if ok]
        mode = "dry run, nothing will be sent" if dry_run else (", ".join(targets) or "no targets configured")
        self.log(f"Messaging Agent is ready ({mode})")

    def targets(self) -> dict:
        return {
            "Pushover": bool(get(self.settings, "notifications.pushover", True) and self.pushover_user and self.pushover_token),
            "Telegram bot": bool(get(self.settings, "notifications.telegram_bot", True) and self.tg_bot_token and self.tg_chat_id),
        }

    def push(self, text):
        """
        Send a notification to every configured target (Pushover and/or Telegram bot)
        """
        self.sent.append(text)
        if self.dry_run:
            self.log("Messaging Agent (dry run) would send: " + text.replace("\n", " | "))
            return
        targets = self.targets()
        if targets["Pushover"]:
            self.log("Messaging Agent is sending a push notification")
            payload = {
                "user": self.pushover_user,
                "token": self.pushover_token,
                "message": text[:1024],
                "sound": "cashregister",
            }
            try:
                requests.post(pushover_url, data=payload, timeout=10)
            except requests.RequestException as exc:
                self.log(f"Pushover failed: {type(exc).__name__}")
        if targets["Telegram bot"]:
            self.log("Messaging Agent is sending a Telegram bot message")
            try:
                requests.post(
                    telegram_url.format(token=self.tg_bot_token),
                    data={"chat_id": self.tg_chat_id, "text": text[:4000]},
                    timeout=10,
                )
            except requests.RequestException as exc:
                # Never log the exception text: it contains the URL with the bot token.
                self.log(f"Telegram bot message failed: {type(exc).__name__}")
        if not any(targets.values()):
            self.log("Messaging Agent has no notification target configured; logged only")

    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        self.push(format_alert(opportunity))
        self.log("Messaging Agent has completed")

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float, currency: str = "USD"
    ) -> str:
        from litellm import completion

        price, value = format_money(deal_price, currency), format_money(estimated_true_value, currency)
        user_prompt = "Please summarize this great deal in 2-3 sentences to be sent as an exciting push notification alerting the user about this deal.\n"
        user_prompt += f"Item Description: {description}\nOffered Price: {price}\nEstimated true value: {value}"
        if currency == "INR":
            user_prompt += "\nPrices are in Indian rupees; write them with the ₹ sign and Indian digit grouping."
        user_prompt += "\n\nRespond only with the 2-3 sentence message which will be used to alert & excite the user about this deal"
        response = completion(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str, currency: str = "USD"):
        """
        Make an alert about the specified details
        """
        self.log("Messaging Agent is using Claude to craft the message")
        text = self.craft_message(description, deal_price, estimated_true_value, currency)
        self.push(text[:200] + "... " + url)
        self.log("Messaging Agent has completed")
