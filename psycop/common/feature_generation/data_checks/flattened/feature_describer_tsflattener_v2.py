from dataclasses import dataclass


@dataclass
class ParsedPredictorColumn:
    col_name: str
    feature_name: str
    fallback: str
    time_interval_start: str
    time_interval_end: str
    resolve_multiple_strategy: str
    is_static: bool


def parse_predictor_column_name(
    col_name: str,
    is_static_regex: str = r"fallback",
    feature_name_regex: str = r"[pred|outc]_(.+)_fallback",
    fallback_regex: str = r"_fallback_(.+)$",
    time_interval_start_regex: str = r"_within_([0-9]+)",
    time_interval_end_regex: str = "_to_([0-9]+)_",
    resolve_multiple_strategy_regex: str = r"([a-zA]+)_fallback",
) -> ParsedPredictorColumn:
    pass
