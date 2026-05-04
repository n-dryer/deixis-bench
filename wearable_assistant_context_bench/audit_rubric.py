"""Rule-based audit rubric for the scenario test bank.

Infers ``change_type``, ``target_context``, and ``difficulty_tier`` from
the script content (turn images, user speech, gold answers,
context_image) without reading the metadata being audited. The
intended caller is ``scripts/audit_scenarios.py``, which writes a diff
CSV comparing audit verdicts against the metadata.

Pure functions; no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable


def _ascii_fold(text: str) -> str:
    """Strip diacritics so 'sauté' matches 'saute' in cue lists.

    Without this, gold answer tokens with accented characters slip past
    the substring tests in ``_SEQUENTIAL_PAIRS`` and similar.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c)
    )

# --- text utilities ---------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase, ASCII-fold, split on non-alphanumerics, return token set."""
    return set(_TOKEN_RE.findall(_ascii_fold(text or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Set Jaccard similarity. Returns 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_overlap_ratio(a: str, b: str) -> float:
    """Character-level similarity via difflib's longest-matching-block ratio.

    Used to gauge whether T1 and T2 image descriptions are
    near-identical (subtle scene contrast → harder scenario).
    """
    if not a and not b:
        return 0.0
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _gold_field(gold: dict | None, key: str) -> list[str]:
    if not gold:
        return []
    value = gold.get(key)
    return value if isinstance(value, list) else []


# --- target_context inference ---------------------------------------------


_PRIOR_DEIXIS_CUES = (
    "before",
    "earlier",
    "minute ago",
    "from before",
    "from earlier",
    "what was that",
    "what was the",
    "the one i had",
    "the one from",
    "yesterday",
    "a while ago",
    "i noticed",
)


def audit_target_context(gold: dict | None, t2_user: str) -> str:
    """Infer ``target_context`` from ``gold`` indicator-list shape, with T2
    user phrasing breaking the tie when both ``current_answers`` and
    ``prior_answers`` are populated.

    Rule order (first match wins):
    1. ``clarify_indicators`` non-empty → ``clarify``
    2. ``abstain_indicators`` non-empty → ``abstain``
    3. only ``current_answers`` non-empty → ``current``
    4. only ``prior_answers`` non-empty → ``prior``
    5. both populated: ``prior`` if T2 user uses past-reference deixis,
       else ``current``
    """
    clarify = _gold_field(gold, "clarify_indicators")
    abstain = _gold_field(gold, "abstain_indicators")
    current = _gold_field(gold, "current_answers")
    prior = _gold_field(gold, "prior_answers")

    if clarify:
        return "clarify"
    if abstain:
        return "abstain"
    if current and not prior:
        return "current"
    if prior and not current:
        return "prior"
    if current and prior:
        text = (t2_user or "").lower()
        if any(cue in text for cue in _PRIOR_DEIXIS_CUES):
            return "prior"
        return "current"
    return "current"


# --- change_type rules ------------------------------------------------------


# Screen_content scenarios in the bank universally describe the screen
# as the focal element with phrases like "backlit display", "screen
# showing", "the display now showing", "compose window", or "speech
# bubbles". Bare "display" / "monitor" appear too often as peripheral
# scene elements (a tuner clip-on display, a rear camera display) to be
# usable cues, so the list below requires more specific phrasing.
_SCREEN_CUES = (
    "backlit",  # canonical descriptor in screen_content scenarios
    "screen showing",
    "screen now",
    "the screen",
    "compose window",
    "speech bubbles",
    "phone screen",
    "tablet screen",
    "smartphone screen",
    "drawing slate",
)

# Strict absent_referent T2 USER cues. The phrases here picked out
# every absent_referent scenario in the bank (sc-38..sc-42, adv-20,
# sc-121..sc-135, sc-126..sc-128). Conservative on purpose: object_state
# scenarios use "this" / "the original X" to ask about the present
# object's prior state; absent_referent scenarios use "that one" /
# "that thing" / "the [printing|label|...]" to ask about a removed
# entity.
_ABSENT_T2_USER_CUES = (
    "the one from",
    "the one before",
    "the earlier piece",
    "the earlier one",
    "that one",
    "those, again",
    "those again",
    "that next turn",
    "that handwritten",
    "that thing",
    "what was that",
    # Specific "the [gone-noun]" patterns
    "the printing",
    "the dosage",
    "the wattage",
    "what torque",
    "the gauge actually",
    "the volume level",
    "the model number",
    "the total load",
    "the planting depth",
    "the instructions say",
    "did the instructions",
    "the sender on",
)

_HAND_CUES = (
    "hand gripping",
    "hand wrapped",
    "hand loosely",
    "hand holding",
    "hands holding",
    "hand on the",
    "right hand",
    "left hand",
    "fingers wrapped",
    "fingers gripping",
    "palm pressed",
)

_VIEW_SHIFT_CUES = (
    "camera now",
    "camera tilted",
    "now tilted",
    "now showing a",
    "now angled",
    "view now toward",
    "attention shifts",
    "now in focus",
    "now framed on",
)

_STATE_CHANGE_CUES = (
    "now bubbling",
    "now boiling",
    "churning",
    "now cooked",
    "now wilted",
    "now dried",
    "now wet",
    "now dusted",
    "now charred",
    "now risen",
    "now collapsed",
    "now glowing",
    "now lit",
    "now uniformly",
    "surface of the liquid",
    "rising bubbles",
    "large bubbles",
)

# Activity-pair vocabulary. Each pair is (prior_activity_word,
# current_activity_word) — distinct activities, not state transitions.
# Excluded on purpose: ("rough", "smooth"), since sanding rough → smooth
# is a state transition within the same activity (object_state, not
# sequential_task). Add new pairs here as the bank grows; pairs are
# substring-matched against joined gold answer text.
_SEQUENTIAL_PAIRS = (
    ("sand", "stain"),
    ("sand", "finish"),
    ("sand", "paint"),
    ("sanding", "stain"),
    ("sanding", "finish"),
    ("chop", "saute"),
    ("chop", "cook"),
    ("dice", "saute"),
    ("dice", "cook"),
    ("dicing", "saute"),
    ("slicing", "saute"),
    ("knead", "bake"),
    ("knead", "shape"),
    ("mix", "bake"),
    ("disassemble", "reassemble"),
    ("loosen", "tighten"),
    ("measure", "cut"),
    ("mark", "cut"),
    ("primer", "paint"),
    ("prep", "cook"),
    ("prune", "tie"),
    ("water", "feed"),
    ("dig", "plant"),
    ("plant", "water"),
    ("rinse", "dry"),
    ("wash", "rinse"),
    ("scrub", "rinse"),
    ("bag", "label"),
    ("charge", "insert"),
    ("charging", "insertion"),
    ("tune", "chord"),
    ("tune", "strum"),
    ("tuning", "fretting"),
    ("tuning", "chord"),
    ("detergent", "transfer"),
    ("wash", "dry"),
    ("washing", "drying"),
    ("sketch", "ink"),
    ("sketching", "inking"),
    ("pencil", "ink"),
    ("loosen", "jack"),
    ("lug", "jack"),
    ("breaker", "jack"),
    ("stretch", "squat"),
    ("stretching", "squat"),
    ("warm-up", "barbell"),
    ("warm-up", "squat"),
    ("trowel", "plant"),
    ("trowel", "seedling"),
    ("rise", "bake"),
    ("proof", "bake"),
    ("rise time", "bake time"),
)


def _starts_same(text: str) -> bool:
    """T2 image opens with a 'Same X' or 'The same X' anchor (case-insensitive).

    Both forms anchor T2 to the T1 scene and rule out a location change.
    """
    head = (text or "").lstrip().lower()
    return head.startswith("same ") or head.startswith("the same ")


def _matches_any(text: str, cues: Iterable[str]) -> str | None:
    """Return the first matching cue, or None. Substring match against an
    ASCII-folded, lowercased text (so cues match 'sauté' as 'saute')."""
    lowered = _ascii_fold(text or "").lower()
    for cue in cues:
        if cue in lowered:
            return cue
    return None


def audit_change_type(scenario: dict) -> tuple[str, list[str]]:
    """Infer ``change_type`` from the script. Returns ``(label, signals)``
    where ``signals`` is the list of rule traces that fired.

    Rules are ordered most-specific first; the first match wins. The
    full trace is preserved so reviewers can see why a verdict was
    reached.
    """
    signals: list[str] = []

    t1i = scenario.get("turn_1_image") or ""
    t2i = scenario.get("turn_2_image") or ""
    t2u = scenario.get("turn_2_user") or ""
    ctx_img = scenario.get("context_image")
    gold = scenario.get("gold") or {}

    # Rule 1: cross_session_reference — context_image is non-null.
    # The schema invariant guarantees this fires iff the scenario was
    # authored as cross_session_reference.
    if ctx_img:
        signals.append("rule_1_cross_session_reference: context_image is non-null")
        return "cross_session_reference", signals

    # Rule 2: screen_content — both turns describe a screen surface.
    t1_screen = _matches_any(t1i, _SCREEN_CUES)
    t2_screen = _matches_any(t2i, _SCREEN_CUES)
    if t1_screen and t2_screen:
        signals.append(
            f"rule_2_screen_content: T1 screen cue {t1_screen!r}, T2 screen cue {t2_screen!r}"
        )
        return "screen_content", signals

    # Rule 3: absent_referent — T2 user uses past-reference deixis on a
    # removed entity. Image cues alone are too noisy ("no longer
    # reflective", "no longer flowing", and similar appear in
    # object_state state transitions), so this rule is anchored on T2
    # user phrasing only.
    t2u_absent = _matches_any(t2u, _ABSENT_T2_USER_CUES)
    if t2u_absent:
        signals.append(f"rule_3_absent_referent: T2 user cue {t2u_absent!r}")
        return "absent_referent", signals

    # Rule 4: object_in_hand — hand cues in BOTH turns.
    # Placed before location so low-Jaccard hand-vs-hand scenarios route
    # correctly. (Hand scenes often share little vocabulary across turns.)
    t1_hand = _matches_any(t1i, _HAND_CUES)
    t2_hand = _matches_any(t2i, _HAND_CUES)
    if t1_hand and t2_hand:
        signals.append(
            f"rule_4_object_in_hand: T1 hand cue {t1_hand!r}, T2 hand cue {t2_hand!r}"
        )
        return "object_in_hand", signals

    # Rule 5: sequential_task — a known activity-transition pair appears
    # across prior→current gold (e.g. sand → stain, chop → saute).
    # Fires regardless of whether T2 starts with 'Same', because the
    # bank includes sequential scenarios where the whole work-station
    # changes (sc-22: cutting board → pan on stove).
    prior_text = " ".join(_gold_field(gold, "prior_answers")).lower()
    current_text = " ".join(_gold_field(gold, "current_answers")).lower()
    for prior_word, current_word in _SEQUENTIAL_PAIRS:
        if prior_word in prior_text and current_word in current_text:
            signals.append(
                f"rule_5_sequential_task: gold pair {prior_word!r} -> {current_word!r}"
            )
            return "sequential_task", signals

    # Rule 6: location — T2 image does NOT open with 'Same'/'The same'
    # anchor AND T1/T2 word-Jaccard is low, meaning the scene vocabulary
    # changed wholesale. Comes after sequential_task so chop → saute
    # (low-Jaccard but still a task transition) routes correctly.
    t1_tokens = tokenize(t1i)
    t2_tokens = tokenize(t2i)
    image_jaccard = jaccard(t1_tokens, t2_tokens)
    if not _starts_same(t2i) and image_jaccard < 0.30:
        signals.append(
            f"rule_6_location: T2 not anchored to T1, image-token Jaccard={image_jaccard:.2f}"
        )
        return "location", signals

    # Rule 7: object_in_view — T2 starts 'Same'/'The same' AND has a
    # camera/attention shift cue.
    view_cue = _matches_any(t2i, _VIEW_SHIFT_CUES)
    if _starts_same(t2i) and view_cue:
        signals.append(f"rule_7_object_in_view: T2 view-shift cue {view_cue!r}")
        return "object_in_view", signals

    # Rule 8: sequential_task — when gold prior/current are both empty
    # (target is clarify or abstain, no specific vocabulary anchored),
    # the discriminating signal is 'Same' + low T1/T2 Jaccard. Empty
    # gold rules out object_state (which always describes prior and
    # current state vocabulary); low Jaccard is consistent with a new
    # tool/operation introduced in T2 (sequential median ≈ 0.20 vs
    # object_in_view ≈ 0.30, object_state ≈ 0.32).
    has_gold_pc = bool(_gold_field(gold, "prior_answers")) or bool(
        _gold_field(gold, "current_answers")
    )
    if _starts_same(t2i) and not has_gold_pc and image_jaccard < 0.25:
        signals.append(
            f"rule_8_sequential_task: 'Same' anchor + empty gold + low Jaccard={image_jaccard:.2f}"
        )
        return "sequential_task", signals

    # Rule 9: object_state — default for 'Same X'/'The same X' that
    # didn't match sequential or view-shift. Most 'Same'-anchored
    # scenarios are state changes (paint drying, dough rising, ink
    # curing). The high-Jaccard case (T2 doesn't start with 'Same' but
    # shares scene vocabulary with T1) also lands here.
    if _starts_same(t2i):
        signals.append("rule_9_object_state: T2 starts 'Same'/'The same' (default)")
        return "object_state", signals
    if image_jaccard >= 0.30:
        signals.append(
            f"rule_9_object_state: image-token Jaccard={image_jaccard:.2f} (shared scene, no anchor)"
        )
        return "object_state", signals

    # Last-resort fallback for unanchored T2 with mid-Jaccard — treat as
    # object_in_view (camera moved, no whole-scene change).
    signals.append(
        f"fallback_object_in_view: no rule fired; image-token Jaccard={image_jaccard:.2f}"
    )
    return "object_in_view", signals


# --- difficulty rubric ------------------------------------------------------


_LONG_TIME_CUES = (
    "yesterday",
    "last week",
    "earlier today",
    "this morning",
    "an hour ago",
    "hours ago",
    "before lunch",
    "before bed",
    "next morning",
    "next day",
)


_DISTRACTOR_T2_CUES = (
    "several",
    "multiple",
    "row of",
    "a few",
    "various",
    "different ones",
    "different ones of",
    "an array of",
    "a cluster of",
    "a set of",
    "distinct",
)


def _is_referent_offscreen(target_context: str) -> bool:
    """Audit-derived offscreen flag.

    The prior referent is offscreen exactly when the model is expected
    to ground in something not currently visible — i.e., target=prior or
    target=abstain. This is a coarse approximation; the bank's authored
    label distinguishes ``referent_offscreen`` from ``single_referent``
    in finer detail, but for difficulty scoring this signal is what
    matters: the model can't just look at T2.
    """
    return target_context in ("prior", "abstain")


def _has_distractor(scenario: dict) -> bool:
    t2i = scenario.get("turn_2_image") or ""
    return _matches_any(t2i, _DISTRACTOR_T2_CUES) is not None


def _has_long_time_gap(t2_user: str) -> bool:
    return _matches_any(t2_user, _LONG_TIME_CUES) is not None


_HEAVY_CHANGE_TYPES = frozenset({
    "cross_session_reference",
    "absent_referent",
    "screen_content",
})


def audit_difficulty(
    scenario: dict,
    *,
    target_context: str,
    change_type: str,
) -> tuple[str, int, dict]:
    """Score difficulty additively from script-derived signals.

    Returns ``(tier, score, breakdown)`` where ``breakdown`` maps each
    signal name to the points it contributed (only signals that fired
    are included).

    Bins:
    - ``easy``: total <= 1
    - ``medium``: 2 <= total <= 3
    - ``hard``: total >= 4
    """
    breakdown: dict[str, int] = {}
    score = 0

    if target_context in ("abstain", "clarify"):
        breakdown["target_context_abstain_or_clarify"] = 2
        score += 2

    if _is_referent_offscreen(target_context):
        breakdown["referent_offscreen"] = 2
        score += 2

    if _has_distractor(scenario):
        breakdown["distractor_present"] = 1
        score += 1

    if change_type in _HEAVY_CHANGE_TYPES:
        breakdown["heavy_change_type"] = 1
        score += 1

    gold = scenario.get("gold") or {}
    prior_tokens = tokenize(" ".join(_gold_field(gold, "prior_answers")))
    current_tokens = tokenize(" ".join(_gold_field(gold, "current_answers")))
    overlap = jaccard(prior_tokens, current_tokens)
    if overlap >= 0.30:
        breakdown["gold_jaccard_high"] = 2
        score += 2
    elif overlap >= 0.10:
        breakdown["gold_jaccard_medium"] = 1
        score += 1

    if _has_long_time_gap(scenario.get("turn_2_user") or ""):
        breakdown["long_time_gap"] = 1
        score += 1

    image_chars = char_overlap_ratio(
        scenario.get("turn_1_image") or "",
        scenario.get("turn_2_image") or "",
    )
    if image_chars >= 0.70:
        breakdown["subtle_scene_contrast"] = 1
        score += 1

    if (
        change_type == "object_in_hand"
        and target_context == "current"
        and not _has_distractor(scenario)
        and not _is_referent_offscreen(target_context)
    ):
        breakdown["object_in_hand_current_single_bonus"] = -1
        score -= 1

    if score <= 1:
        tier = "easy"
    elif score <= 3:
        tier = "medium"
    else:
        tier = "hard"

    return tier, score, breakdown
