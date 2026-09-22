"""Experimental evaluation and benchmarking package."""

from .compare_methods import compare_methods, compare_methods_from_dataframe, summarize_method_results
from .failure_analysis import classify_failure, group_failure_reasons
from .logger import TrialLogger, aggregate_trials, load_trial_results

__all__ = [
    "TrialLogger",
    "aggregate_trials",
    "load_trial_results",
    "compare_methods",
    "compare_methods_from_dataframe",
    "summarize_method_results",
    "classify_failure",
    "group_failure_reasons",
]
