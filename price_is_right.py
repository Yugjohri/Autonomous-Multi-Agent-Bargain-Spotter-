import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv

load_dotenv(override=True)

if "--fixture" in sys.argv:
    # Must be set before gradio is imported, so fixture runs make no network calls.
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from agents.deals import Opportunity  # noqa: E402
from agents.money import format_money  # noqa: E402
from agents.normalize import STORE_NAMES  # noqa: E402
from deal_agent_framework import DealAgentFramework  # noqa: E402
from log_utils import reformat  # noqa: E402

ALL_STORES = "All stores"
HEADERS = ["Deal", "Price", "MRP", "Estimate", "Discount", "Discount %", "Confidence", "Store", "Source", "Age", "URL"]


class LogBuffer(logging.Handler):
    """The most recent log lines, from every run (scheduled, live or manual), for the UI."""

    def __init__(self, size: int = 200):
        super().__init__()
        self.lines = deque(maxlen=size)
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"))

    def emit(self, record):
        self.lines.append(reformat(self.format(record)))

    def html(self, last: int = 18) -> str:
        return html_for(list(self.lines)[-last:])


def html_for(log_lines):
    output = "<br>".join(log_lines)
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def age_label(posted_at) -> str:
    if posted_at is None:
        return ""
    posted = posted_at if posted_at.tzinfo else posted_at.replace(tzinfo=timezone.utc)
    minutes = int((datetime.now(timezone.utc) - posted).total_seconds() // 60)
    if minutes < 60:
        return f"{max(minutes, 0)} min"
    if minutes < 48 * 60:
        return f"{minutes // 60} h"
    return f"{minutes // 1440} d"


def row_for(opp: Opportunity) -> list:
    deal, currency = opp.deal, opp.currency or opp.deal.currency
    source = deal.source + (f" +{len(deal.seen_in) - 1}" if len(deal.seen_in) > 1 else "")
    return [
        deal.title or deal.product_description[:90],
        format_money(deal.price, currency),
        format_money(deal.mrp, currency) if deal.mrp else "",
        format_money(opp.estimate, currency),
        format_money(opp.discount, currency),
        f"{opp.discount_pct:.0f}%" if opp.discount_pct is not None else "",
        opp.confidence or "",
        STORE_NAMES.get(deal.store, deal.store),
        source,
        age_label(deal.posted_at),
        deal.url,
    ]


def table_for(opps: List[Opportunity], store: str = ALL_STORES, currency: str = None) -> list:
    """Newest first, optionally only one store and one currency (INR mode hides old USD deals)."""
    rows = [
        o for o in reversed(opps)
        if (currency is None or (o.currency or o.deal.currency) == currency)
        and (store in (None, ALL_STORES) or STORE_NAMES.get(o.deal.store, o.deal.store) == store)
    ]
    return [row_for(o) for o in rows]


class App:
    def __init__(self, fixture: bool = False, dry_run: bool = False):
        self.fixture = fixture
        self.dry_run = dry_run
        self.agent_framework = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework(fixture=self.fixture, dry_run=self.dry_run)
        return self.agent_framework

    @property
    def currency(self) -> str:
        return "USD" if self.get_agent_framework().mode == "usd_legacy" else "INR"

    def table(self, store: str = ALL_STORES) -> list:
        return table_for(self.get_agent_framework().memory, store, self.currency)

    def store_choices(self) -> list:
        opps = [o for o in self.get_agent_framework().memory if (o.currency or o.deal.currency) == self.currency]
        stores = sorted({STORE_NAMES.get(o.deal.store, o.deal.store) for o in opps})
        return [ALL_STORES] + [s for s in stores if s]

    def send_alert(self, url: str) -> str:
        """Send one alert for the deal with this URL (the user asked for it explicitly)."""
        if not url:
            return "Select a deal in the table first."
        for opp in self.get_agent_framework().memory:
            if opp.deal.url == url:
                self.get_agent_framework().init_agents_as_needed()
                sent = self.get_agent_framework().planner.messenger.alert(opp, force=True)
                title = opp.deal.title or opp.deal.product_description[:60]
                return f"Alert {'sent' if sent else 'not sent'}: {title}"
        return "That deal is no longer in memory."

    def run(self):
        framework = self.get_agent_framework()
        mode = framework.mode
        if self.fixture:
            subtitle = "Fixture mode: replaying saved posts offline. No network calls, no notifications."
        elif self.dry_run:
            subtitle = "Dry run: live sources, no notifications."
        else:
            subtitle = "Telegram deal channels and Indian deal sites, valued in rupees by GPT with INR price context."
        if mode == "usd_legacy":
            subtitle = "Legacy US mode: DealNews RSS valued by a fine-tuned Llama, GPT with RAG and a neural network."

        log_buffer = LogBuffer()
        logging.getLogger().addHandler(log_buffer)

        with gr.Blocks(title="Bargain Spotter", fill_width=True) as ui:

            def get_plot():
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800, mode=mode)
                if len(documents) < 5:
                    fig = go.Figure()
                    fig.update_layout(title="Not enough products in the vector store yet", height=400)
                    return fig
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            z=vectors[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=colors, opacity=0.7),
                            text=[d[:80] for d in documents],
                            hoverinfo="text",
                        )
                    ]
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),
                        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
                    ),
                    height=400,
                    margin=dict(r=5, b=1, l=5, t=2),
                )
                return fig

            def scan_now():
                started = self.get_agent_framework().scan_now()
                return "Scan started; watch the log below." if started else "A scan is already running."

            def refresh(store):
                return self.table(store), gr.update(choices=self.store_choices())

            def do_select(evt: gr.SelectData):
                # Selecting a row only remembers it. Gradio fires this for every cell click (and
                # again when the table refreshes), so it must never send anything by itself.
                row = getattr(evt, "row_value", None)
                url = row[-1] if row else None
                title = row[0] if row else ""
                return url, f"Selected: {title}" if url else "Select a deal in the table first."

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px"><strong>Bargain Spotter</strong> - autonomous agents hunting Indian deals in rupees</div>'
                )
            with gr.Row():
                gr.Markdown(f'<div style="text-align: center;font-size:14px">{subtitle}</div>')
            with gr.Row():
                store_filter = gr.Dropdown(choices=self.store_choices(), value=ALL_STORES, label="Store", scale=0, min_width=220)
                alert_button = gr.Button("Send alert for selected deal", scale=0, min_width=240)
                scan_button = gr.Button("Scan now", scale=0, min_width=120)
                alert_status = gr.Markdown("")
            selected_url = gr.State(None)
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=HEADERS,
                    wrap=True,
                    column_widths=[5, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3],
                    row_count=10,
                    col_count=len(HEADERS),
                    max_height=420,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(value=get_plot(), show_label=False)

            # Scans run on the server (DealAgentFramework.start_scheduler); the page only displays.
            ui.load(log_buffer.html, outputs=[logs])
            ui.load(self.table, inputs=[store_filter], outputs=[opportunities_dataframe])
            log_timer = gr.Timer(value=3, active=True)
            log_timer.tick(log_buffer.html, outputs=[logs])
            refresh_timer = gr.Timer(value=15, active=True)
            refresh_timer.tick(refresh, inputs=[store_filter], outputs=[opportunities_dataframe, store_filter])
            scan_button.click(scan_now, outputs=[alert_status])
            store_filter.change(self.table, inputs=[store_filter], outputs=[opportunities_dataframe])
            opportunities_dataframe.select(do_select, outputs=[selected_url, alert_status])
            alert_button.click(self.send_alert, inputs=[selected_url], outputs=[alert_status])

        if framework.start_live():
            logging.info("Live Telegram updates are on")
        framework.start_scheduler()
        ui.launch(share=False, inbrowser=os.getenv("NO_BROWSER") != "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Bargain Spotter: autonomous agents hunting Indian deals")
    parser.add_argument("--fixture", action="store_true", help="Replay saved posts offline; no network, no notifications")
    parser.add_argument("--dry-run", action="store_true", help="Run the live pipeline but send no notifications")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    App(fixture=args.fixture, dry_run=args.dry_run).run()
