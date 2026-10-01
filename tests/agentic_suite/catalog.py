"""Shared tool schemas, call builders and the sample catalogue for the agentic suite.

The offline tests use the builders to write hand-computed cases; the live tests
run ``LIVE_CASES`` against real models. Every live case says what kind of
behaviour it probes so the report can group results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}

TOOLS = [
    fn("get_weather", "Current weather for a city.",
       {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("get_time", "Current local time in a city.", {"city": S}, ["city"]),
    fn("get_stock_price", "Latest stock price for a ticker symbol, e.g. AAPL.", {"ticker": S}, ["ticker"]),
    fn("convert_currency", "Convert an amount between currencies given as ISO 4217 codes.",
       {"amount": {"type": "number"}, "from_currency": S, "to_currency": S},
       ["amount", "from_currency", "to_currency"]),
    fn("search_flights", "Search flights by IATA airport codes and date (YYYY-MM-DD).",
       {"origin": S, "destination": S, "date": S}, ["origin", "destination", "date"]),
    fn("book_flight", "Book a flight by the id returned from search_flights.", {"flight_id": S}, ["flight_id"]),
    fn("create_event", "Create a calendar event.",
       {"title": S, "date": S, "attendees": {"type": "array", "items": S},
        "reminder": {"type": "object", "properties": {"minutes_before": {"type": "integer"}}}},
       ["title", "date"]),
    fn("set_alarm", "Set an alarm.", {"time": S, "enabled": {"type": "boolean"},
                                      "repeat_days": {"type": "integer"}}, ["time"]),
]

# Distractors: plausible tools that should NOT be chosen for the live cases.
DISTRACTORS = [
    fn("get_news", "Latest news headlines for a topic.", {"topic": S}, ["topic"]),
    fn("translate", "Translate text into a target language.", {"text": S, "target_language": S},
       ["text", "target_language"]),
    fn("get_air_quality", "Air quality index for a city.", {"city": S}, ["city"]),
    fn("get_crypto_price", "Latest price of a cryptocurrency, e.g. BTC.", {"symbol": S}, ["symbol"]),
    fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
]


def c(name: str, **args: Any) -> dict:
    return {"name": name, "arguments": args}


def W(city: str, **kw: Any) -> dict:
    return c("get_weather", city=city, **kw)


def T(city: str) -> dict:
    return c("get_time", city=city)


def STOCK(t: str) -> dict:
    return c("get_stock_price", ticker=t)


BOOK = c("book_flight", flight_id="AZ-61")
SEARCH = c("search_flights", origin="DEL", destination="BOM", date="2026-10-01")


@dataclass
class LiveCase:
    """One live sample: prompt, reference NEXT turn, and what it probes."""

    id: str
    probe: str
    input: str
    expected: list
    tools: list = field(default_factory=lambda: TOOLS)
    # Metrics that must produce a score for this case whatever the model does
    # (a metric can legitimately be omitted when it has no denominator).
    note: str = ""


LIVE_CASES = [
    LiveCase("single", "single call", "What's the weather in Tokyo right now?", [[W("Tokyo")]]),
    LiveCase("single-enum", "enum argument", "What's the weather in Chicago in fahrenheit?",
             [[W("Chicago", unit="fahrenheit")]]),
    LiveCase("single-numeric", "number argument", "Convert 250 USD to EUR.",
             [[c("convert_currency", amount=250, from_currency="USD", to_currency="EUR")]]),
    LiveCase("par-2-same", "parallel, same tool x2", "Compare the current weather in Paris and Rome.",
             [[W("Paris"), W("Rome")]]),
    LiveCase("par-3-same", "parallel, same tool x3", "Get the latest stock prices for AAPL, MSFT and NVDA.",
             [[STOCK("AAPL"), STOCK("MSFT"), STOCK("NVDA")]]),
    LiveCase("par-2-mixed", "parallel, two tools same entity", "What's the weather and the local time in London?",
             [[W("London"), T("London")]]),
    LiveCase("par-2-unrelated", "parallel, two unrelated tools",
             "What's the weather in Berlin, and how much is 50 GBP in INR?",
             [[W("Berlin"), c("convert_currency", amount=50, from_currency="GBP", to_currency="INR")]]),
    LiveCase("par-4-grid", "parallel, 2 tools x 2 cities", "Tell me the weather and local time in both Madrid and Lisbon.",
             [[W("Madrid"), T("Madrid"), W("Lisbon"), T("Lisbon")]]),
    LiveCase("dependent", "dependent chain (next turn only)",
             "Find flights from DEL to BOM on 2026-10-01 and book the cheapest one.", [[SEARCH]],
             note="booking needs the search result, so only search belongs in the first turn"),
    LiveCase("irrelevant-math", "irrelevance (no tool fits)", "What is 17 multiplied by 3?", []),
    LiveCase("irrelevant-chat", "irrelevance (chit-chat)", "Thanks, that's all for today!", []),
    LiveCase("nested-args", "array + object arguments",
             "Create a calendar event titled 'Design review' on 2026-10-05 with alice@x.com and bob@x.com, "
             "remind me 15 minutes before.",
             [[c("create_event", title="Design review", date="2026-10-05",
                 attendees=["alice@x.com", "bob@x.com"], reminder={"minutes_before": 15})]]),
    LiveCase("boolean-arg", "boolean + integer arguments", "Set an alarm for 07:30, enabled, repeating 5 days.",
             [[c("set_alarm", time="07:30", enabled=True, repeat_days=5)]]),
    LiveCase("distractors", "many plausible distractor tools", "What's the air temperature in Oslo right now?",
             [[W("Oslo")]], tools=TOOLS + DISTRACTORS),
    LiveCase("non-english", "non-English prompt", "Quel temps fait-il à Marseille et à Nice en ce moment ?",
             [[W("Marseille"), W("Nice")]]),
    LiveCase("noisy-long", "long noisy prompt",
             "So I've been planning this trip for months, my sister keeps telling me to pack light, and honestly "
             "I still haven't decided about the hotel, but anyway, before any of that I just need to know one "
             "thing: what's the local time in Singapore at the moment?", [[T("Singapore")]]),
]

# Multi-turn cases for the live agent loop (the agent executes stub tools).
# The stubs make the right answer deterministic: Rome is warmer; UK-102 is cheapest.
AGENT_CASES = [
    LiveCase("agent-par-then-dep", "parallel group then dependent call",
             "Which is warmer right now, Paris or Rome? Then tell me the local time in the warmer city.",
             [[W("Paris"), W("Rome")], [T("Rome")]]),
    LiveCase("agent-search-book", "search then book (dependent)",
             "Find flights from DEL to BOM on 2026-10-01 and book the cheapest one.",
             [[SEARCH], [c("book_flight", flight_id="UK-102")]]),
    LiveCase("agent-single", "one call then answer", "What's the weather in Tokyo?", [[W("Tokyo")]]),
    LiveCase("agent-irrelevant", "no tool needed", "What is 12 + 30?", []),
]
