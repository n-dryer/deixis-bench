"""Tests for the scenario audit rubric.

The rubric infers ``change_type`` and ``difficulty_tier`` from the
script content alone (turn_*_image, turn_*_user, gold, context_image)
without consulting the metadata being audited. These tests exercise
each of the eight ``change_type`` rules with a minimal fixture, plus
target-context inference and difficulty scoring edges.
"""

from __future__ import annotations

import pytest

from wearable_assistant_context_bench.audit_rubric import (
    audit_change_type,
    audit_difficulty,
    audit_target_context,
    char_overlap_ratio,
    jaccard,
    tokenize,
)


# --- target_context inference ---------------------------------------


def test_target_context_clarify_when_clarify_indicators_present():
    gold = {
        "current_answers": [],
        "prior_answers": ["a"] * 8,
        "clarify_indicators": ["which"] * 10,
        "abstain_indicators": [],
    }
    assert audit_target_context(gold, t2_user="Is this the right one?") == "clarify"


def test_target_context_abstain_when_abstain_indicators_present_and_no_current():
    gold = {
        "current_answers": [],
        "prior_answers": ["a"] * 9,
        "clarify_indicators": [],
        "abstain_indicators": ["can't tell"] * 8,
    }
    assert audit_target_context(gold, t2_user="What did the printing say?") == "abstain"


def test_target_context_current_when_current_answers_only():
    gold = {
        "current_answers": ["pot", "boiling"] + ["x"] * 6,
        "prior_answers": [],
        "clarify_indicators": [],
        "abstain_indicators": [],
    }
    assert audit_target_context(gold, t2_user="Can I add the noodles yet?") == "current"


def test_target_context_prior_when_only_prior_answers():
    gold = {
        "current_answers": [],
        "prior_answers": ["goldfinch"] + ["y"] * 7,
        "clarify_indicators": [],
        "abstain_indicators": [],
    }
    assert audit_target_context(gold, t2_user="What was that one earlier?") == "prior"


# --- change_type rule fires ---------------------------------------


def _gold(current=None, prior=None, clarify=None, abstain=None):
    return {
        "current_answers": current or [],
        "prior_answers": prior or [],
        "clarify_indicators": clarify or [],
        "abstain_indicators": abstain or [],
    }


def test_cross_session_reference_fires_when_context_image_present():
    sc = {
        "scenario_id": "fx-csr",
        "context_image": "Earlier scene of a fridge interior.",
        "turn_1_image": "Open fridge view, top shelf items.",
        "turn_1_user": "What should I grab first to make breakfast?",
        "turn_2_image": "Kitchen counter with cracked eggshells on a plate.",
        "turn_2_user": "Hey, what was that note I noticed before I opened the fridge?",
        "gold": _gold(current=["plate"] * 7, prior=["sticky note"] * 8),
    }
    label, signals = audit_change_type(sc)
    assert label == "cross_session_reference"
    assert any("context_image" in s for s in signals)


def test_screen_content_fires_when_both_turns_show_a_screen():
    sc = {
        "scenario_id": "fx-screen",
        "context_image": None,
        "turn_1_image": "Handheld backlit rectangular display showing speech bubbles.",
        "turn_1_user": "How should I respond to this?",
        "turn_2_image": "Same handheld backlit rectangular display, now a compose window with a subject line.",
        "turn_2_user": "How should I respond to this?",
        "gold": _gold(current=["email", "subject"] * 4, prior=["text", "thread"] * 4),
    }
    label, signals = audit_change_type(sc)
    assert label == "screen_content"


def test_absent_referent_fires_when_t2_describes_referent_gone():
    sc = {
        "scenario_id": "fx-absent",
        "context_image": None,
        "turn_1_image": "Cylindrical clear plastic tube hanging from a maple branch with a small bird perched on the lowest perch.",
        "turn_1_user": "Cute. Is that a regular for this season?",
        "turn_2_image": "Same maple branch, same hanging tube. Lowest perch is empty. A scatter of light husks on the patio stones below.",
        "turn_2_user": "What was the one from a minute ago, do you think?",
        "gold": _gold(current=["empty feeder"] * 7, prior=["goldfinch"] * 8),
    }
    label, signals = audit_change_type(sc)
    assert label == "absent_referent"


def test_location_fires_when_t2_opens_different_setting():
    sc = {
        "scenario_id": "fx-location",
        "context_image": None,
        "turn_1_image": "Wide view of a low-ceiling room with soft carpet and an upholstered platform along one wall.",
        "turn_1_user": "Should I open the window in here for a bit?",
        "turn_2_image": "Wide view of a smaller room with bright fluorescent lighting and glossy tiled walls in pale blue.",
        "turn_2_user": "Should I open the window in here for a bit?",
        "gold": _gold(current=["bathroom"] * 7, prior=["bedroom"] * 8),
    }
    label, _ = audit_change_type(sc)
    assert label == "location"


def test_sequential_task_fires_when_t2_starts_same_surface_with_new_step():
    sc = {
        "scenario_id": "fx-seq",
        "context_image": None,
        "turn_1_image": "Rectangular pine board on two sawhorses. A gritty paper block sliding across the surface in a hand.",
        "turn_1_user": "Should I keep going over this same area?",
        "turn_2_image": "Same pine board on the same sawhorses, surface now uniformly even and dust-free. A flat row of stiff bristles on a wooden handle is dipping into a tin of liquid.",
        "turn_2_user": "Am I doing this evenly enough?",
        "gold": _gold(current=["stain", "brush"] * 4, prior=["sanding", "sandpaper"] * 4),
    }
    label, _ = audit_change_type(sc)
    assert label == "sequential_task"


def test_object_in_view_fires_when_camera_tilts_within_same_scene():
    sc = {
        "scenario_id": "fx-view",
        "context_image": None,
        "turn_1_image": "Close view of a small round fruit on a wooden slab, deep yellow-green skin with slight give.",
        "turn_1_user": "Is this ripe enough yet?",
        "turn_2_image": "Same flat raised wooden slab, camera now tilted toward an oblong fruit lying on it. Bright yellow skin freckled with brown.",
        "turn_2_user": "Is this ripe enough yet?",
        "gold": _gold(current=["banana"] * 7, prior=["avocado"] * 8),
    }
    label, _ = audit_change_type(sc)
    assert label == "object_in_view"


def test_object_state_fires_when_same_object_changes_state():
    sc = {
        "scenario_id": "fx-state",
        "context_image": None,
        "turn_1_image": "Tall stainless steel cylindrical vessel on a black glass cooktop, clear liquid filling two-thirds.",
        "turn_1_user": "How long should this take?",
        "turn_2_image": "Tall stainless steel cylindrical vessel with a thin handle on the same cooktop. Surface of the liquid churning with large bubbles.",
        "turn_2_user": "Can I add the noodles yet?",
        "gold": _gold(current=["pot", "boiling"] * 4, prior=["kettle", "heating"] * 4),
    }
    label, _ = audit_change_type(sc)
    assert label == "object_state"


def test_object_in_hand_fires_when_hand_grasps_different_objects():
    sc = {
        "scenario_id": "fx-hand",
        "context_image": None,
        "turn_1_image": "Right hand loosely gripping a long cylindrical wooden shaft with a bowl-shaped wooden head dipped into a steel vessel.",
        "turn_1_user": "Should I keep moving this around or let it sit?",
        "turn_2_image": "Right hand wrapped around a black molded handle attached to a long flat triangular blade resting on a flat board.",
        "turn_2_user": "Is my grip okay for this?",
        "gold": _gold(current=["grip"] * 7, prior=["stir"] * 8),
    }
    label, _ = audit_change_type(sc)
    assert label == "object_in_hand"


# --- difficulty scoring ---------------------------------------


def test_difficulty_easy_for_simple_object_in_hand_current():
    sc = {
        "scenario_id": "fx-easy",
        "context_image": None,
        "turn_1_image": "Right hand gripping a long cylindrical wooden shaft.",
        "turn_1_user": "Should I keep moving this around or let it sit?",
        "turn_2_image": "Right hand wrapped around a black molded handle on a flat blade resting on a board.",
        "turn_2_user": "Is my grip okay for this?",
        "gold": _gold(
            current=["chef's knife", "knife", "pinch grip", "claw the onion", "knuckles tucked", "tip down", "rocking motion"],
            prior=["wooden spoon", "spoon", "stir gently", "scrape the bottom", "fold the sauce", "in a circle", "off the heat"],
        ),
    }
    tier, score, breakdown = audit_difficulty(sc, target_context="current", change_type="object_in_hand")
    assert tier == "easy"
    assert score <= 1


def test_difficulty_hard_for_abstain_with_high_overlap():
    sc = {
        "scenario_id": "fx-hard",
        "context_image": None,
        "turn_1_image": "Generic kitchen scene with a rectangular item on a counter.",
        "turn_1_user": "Should I just toss this or is it still okay?",
        "turn_2_image": "Generic kitchen scene with an empty counter.",
        "turn_2_user": "Wait, what did the printing on the side actually say?",
        "gold": _gold(
            prior=["jar of pickles", "shelf-stable", "expiration", "use by", "best by", "keep refrigerated", "after opening", "label"],
            abstain=["unable to read", "too far", "out of frame", "can't tell", "image is gone", "no longer visible", "insufficient", "unclear"],
        ),
    }
    tier, score, breakdown = audit_difficulty(sc, target_context="abstain", change_type="absent_referent")
    assert tier == "hard"
    assert score >= 4


def test_difficulty_medium_for_clarify_with_low_overlap():
    sc = {
        "scenario_id": "fx-medium",
        "context_image": None,
        "turn_1_image": "Workshop bench with several distinct tools laid out.",
        "turn_1_user": "Am I angling this right?",
        "turn_2_image": "Same bench, attention shifts to a different one of the tools.",
        "turn_2_user": "Is this the right one for what I'm doing?",
        "gold": _gold(
            prior=["mortise chisel", "bevel down", "tap with mallet", "across the grain", "score the line", "register the back", "shoulder cut", "score and pare"],
            clarify=["which one", "which tool", "you mean the one on the left", "I'm not sure which", "can you be more specific", "do you mean the chisel", "which of the chisels", "bevel-edge or mortise", "the one in your hand", "the one on the bench"],
        ),
    }
    tier, score, breakdown = audit_difficulty(sc, target_context="clarify", change_type="object_in_hand")
    assert tier == "medium"
    assert 2 <= score <= 3


# --- utilities ---------------------------------------


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Chef's knife, pinch grip!") == {"chef", "s", "knife", "pinch", "grip"}


def test_jaccard_zero_for_disjoint_sets():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_one_for_identical_sets():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_handles_empty():
    assert jaccard(set(), set()) == 0.0


def test_char_overlap_high_for_near_identical_strings():
    a = "Tall stainless steel cylindrical vessel on a black glass cooktop"
    b = "Tall stainless steel cylindrical vessel on the same cooktop"
    assert char_overlap_ratio(a, b) >= 0.5


def test_char_overlap_low_for_unrelated_strings():
    assert char_overlap_ratio("a wide low-ceiling room with carpet", "a smaller room with tiled walls and fluorescent lighting") < 0.5


# --- end-to-end on a real scenario from the bank ---------------------------------------


def test_real_scenario_sc_01_object_in_hand_current_easy(scenarios_by_id):
    """sc-01 is the canonical object_in_hand example: hand+spoon → hand+knife,
    target=current, single referent. Audit must agree with metadata."""
    sc = scenarios_by_id["sc-01"]
    label, _ = audit_change_type(sc)
    assert label == "object_in_hand"
    target = audit_target_context(sc["gold"], t2_user=sc.get("turn_2_user", ""))
    assert target == "current"
    tier, _, _ = audit_difficulty(sc, target_context=target, change_type=label)
    assert tier == "easy"


def test_real_scenario_sc_47_cross_session_reference(scenarios_by_id):
    """sc-47 is the canonical cross_session_reference example: context_image
    populated, T2 user references pre-T1 state."""
    sc = scenarios_by_id["sc-47"]
    label, _ = audit_change_type(sc)
    assert label == "cross_session_reference"
