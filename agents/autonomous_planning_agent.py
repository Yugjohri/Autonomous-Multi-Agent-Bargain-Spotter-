from datetime import datetime, timezone
from typing import Optional, List, Dict
from agents.agent import Agent
from agents.config import load_settings
from agents.deals import Deal, Opportunity
from agents.money import format_money
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
import json


class AutonomousPlanningAgent(Agent):
    """
    A tool-calling variant of the Planning Agent: GPT drives the workflow itself by
    calling tools to scan, value and notify, and Claude writes the alert text.
    Deals are referred to by number, so the model never has to copy prices or URLs.
    """

    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-5.1"

    def __init__(self, collection=None, settings: Optional[dict] = None, scanner=None, ensemble=None,
                 messenger=None, client=None, dry_run: bool = False):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Autonomous Planning Agent is initializing")
        self.settings = settings if settings is not None else load_settings()
        self.scanner = scanner or ScannerAgent(settings=self.settings)
        self.ensemble = ensemble or EnsembleAgent(collection, settings=self.settings)
        self.messenger = messenger or MessagingAgent(settings=self.settings, dry_run=dry_run)
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.openai = client
        self.memory = None
        self.deals: List[Deal] = []
        self.valuations: Dict[int, object] = {}
        self.opportunity = None
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the tool to scan
        """
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        self.deals = list(results.deals) if results else []
        if not self.deals:
            return "No deals found"
        listing = [
            {
                "deal_number": i,
                "title": d.title,
                "description": d.product_description,
                "price": format_money(d.price, d.currency),
                "mrp": format_money(d.mrp, d.currency) if d.mrp else None,
                "store": d.store,
                "conditions": d.coupon_note,
            }
            for i, d in enumerate(self.deals)
        ]
        return json.dumps(listing, ensure_ascii=False)

    def estimate_true_value(self, deal_number: int) -> str:
        """
        Run the tool to estimate true value
        """
        if not 0 <= deal_number < len(self.deals):
            return "Unknown deal_number"
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        deal = self.deals[deal_number]
        valuation = self.ensemble.value(deal)
        self.valuations[deal_number] = valuation
        return (
            f"Deal {deal_number}: offered at {format_money(deal.price, deal.currency)}, estimated typical price "
            f"{format_money(valuation.estimate, deal.currency)}, {valuation.discount_pct:.0f}% below, "
            f"{valuation.confidence} confidence ({valuation.reason})"
        )

    def notify_user_of_deal(self, deal_number: int) -> str:
        """
        Run the tool to notify the user
        """
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
            return "Already notified; only one notification is allowed"
        if deal_number not in self.valuations:
            return "Estimate this deal's value before notifying"
        deal = self.deals[deal_number]
        valuation = self.valuations[deal_number]
        self.log("Autonomous Planning agent is notifying user")
        self.messenger.notify(deal.title or deal.product_description, deal.price, valuation.estimate, deal.url, deal.currency)
        self.opportunity = Opportunity(
            deal=deal,
            estimate=valuation.estimate,
            discount=valuation.discount,
            discount_pct=valuation.discount_pct,
            confidence=valuation.confidence,
            confidence_reason=valuation.reason,
            found_at=datetime.now(timezone.utc),
        )
        return "Notification sent ok"

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns today's best bargains from Indian deal channels and sites, numbered, with the price in Indian rupees (INR) and any coupon or bank-offer conditions",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Estimate the typical selling price in India (INR) of one scanned deal, with how far below it the deal is and a confidence label",
        "parameters": {
            "type": "object",
            "properties": {
                "deal_number": {
                    "type": "integer",
                    "description": "The deal_number from the scan results",
                },
            },
            "required": ["deal_number"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time, and only for a deal you have estimated",
        "parameters": {
            "type": "object",
            "properties": {
                "deal_number": {
                    "type": "integer",
                    "description": "The deal_number of the deal to notify about",
                },
            },
            "required": ["deal_number"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        """
        Return the json for the tools to be used
        """
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        """
        Actually call the tools associated with this message
        """
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    system_message = (
        "You find great deals on products sold in India using your tools, and notify the user of the best bargain. "
        "All prices are in Indian rupees (INR); never convert currencies."
    )
    user_message = """
    First, use your tool to scan for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal: a large discount below the typical Indian price, preferring higher confidence,
    and use your tool to notify the user about it. If no deal is at least 20% below its typical price, notify nobody.
    Then just reply OK to indicate success.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[Opportunity] = []) -> Optional[Opportunity]:
        """
        Run the full workflow, providing the LLM with tools to surface scraped deals to the user
        :param memory: Opportunities surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.deals, self.valuations = [], {}
        self.opportunity = None
        messages = self.messages[:]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=self.MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
