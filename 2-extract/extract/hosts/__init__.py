"""Host rules, one module per site family (each exports RULES: dict[str, Rule]). Modules are owned
independently, so a host key must appear in only one module."""
from . import vi_health, vi_news, zh_kb, zh_qa

RULES = {}
for _m in (zh_qa, zh_kb, vi_health, vi_news):
    _dup = RULES.keys() & _m.RULES.keys()
    assert not _dup, f"host rule defined twice: {_dup}"
    RULES.update(_m.RULES)
