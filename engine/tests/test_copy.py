from voice_engine import copy

KEYS = [
    "redirect_line", "goodbye_line",
    "devpage.connect", "devpage.disconnect", "devpage.config_editor",
    "devpage.event_log", "devpage.metrics",
    "errors.token_invalid", "errors.session_not_found",
    "errors.session_busy", "errors.upstream_failed",
]


def test_all_keys_present():
    for key in KEYS:
        value = copy.get(key)
        assert isinstance(value, str) and value


def test_backchannel_bank_size():
    bank = copy.backchannel_phrases()
    assert set(bank) == {"neutral", "interested", "amused", "sympathetic"}
    assert all(bank[m] for m in bank)
    assert 15 <= sum(len(v) for v in bank.values()) <= 25
