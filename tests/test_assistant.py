import pytest

from app.assistant import detect_intent


@pytest.mark.parametrize("q, intent", [
    ("hi", "help"),
    ("compare TCS vs INFY", "compare"),
    ("which stocks are rising on low delivery?", "watch"),
    ("who is in a buyer zone today", "zones_list"),
    ("top high conviction stocks", "top_high"),
    ("lowest conviction stocks", "top_low"),
    ("what is delivery %", "definition"),
    ("does this method work?", "evidence"),
    ("show me the backtest", "evidence"),
    ("how is the score calculated", "method"),
    ("is there a buyer zone?", "zone"),
    ("past sharp rises", "history"),
    ("why this score?", "explain"),
    ("summarize this stock", "summary"),
    ("RELIANCE", "summary"),
])
def test_detect_intent(q, intent):
    assert detect_intent(q) == intent
