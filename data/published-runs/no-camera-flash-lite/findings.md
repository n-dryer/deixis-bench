# Wearable Assistant Context Bench: Findings

**Benchmark:** situated context-tracking benchmark for AI wearable assistants used actively for advice or coaching

## Benchmark summary

- **Benchmark**: situated context-tracking benchmark for AI wearable assistants used actively for advice or coaching
- **Default comparison condition**: `baseline`
- **Primary score** - `mean(current_recall, prior_recall)` (class recall, not overall accuracy): **17.0% (95% CI 9.2%-24.8%)**
- **Bootstrap 95% CI on primary score**: 17.0% (95% CI 9.7%-25.4%) (5000 percentile bootstrap iterations)
- **How to read this run**: compare candidate models on the `baseline` score below; treat the other conditions as diagnostic sensitivity checks. CIs are 95% Wilson per class and 95% normal-approximation on the mean recall (with a bootstrap second opinion).
- **Per-class recall under `baseline`** (TP / (TP + FN)):
    - `current_recall`: 6.5% (95% CI 2.2%-17.5%)
    - `prior_recall`: 27.5% (95% CI 16.1%-42.8%)

Condition sensitivity (mean per-class recall):

| Condition | Mean recall (95% CI) |
| --- | --- |
| baseline (default) | 17.0% (95% CI 9.2%-24.8%) |
| context_selection_instruction | 29.3% (95% CI 19.7%-38.9%) |
| pre_answer_context_scaffold | 72.5% (95% CI 64.5%-80.5%) |

## Per task-set recall

| Task set | Mean recall (95% CI) |
| --- | --- |
| `main` | 17.0% (95% CI 9.2%-24.8%) |

## Per shift-type recall

| Shift type | Pass rate (95% CI) |
| --- | --- |
| `absent_referent` | 57.1% (95% CI 36.5%-75.5%) |
| `cross_session_reference` | 50.0% (95% CI 29.9%-70.1%) |
| `location` | 45.5% (95% CI 26.9%-65.3%) |
| `object_in_hand` | 23.8% (95% CI 10.6%-45.1%) |
| `object_in_view` | 52.4% (95% CI 32.4%-71.7%) |
| `object_state` | 45.5% (95% CI 26.9%-65.3%) |
| `screen_content` | 57.1% (95% CI 36.5%-75.5%) |
| `sequential_task` | 27.8% (95% CI 12.5%-50.9%) |

## Per-class pass rate by condition

| Class | baseline | context_selection_instruction | pre_answer_context_scaffold |
| --- | --- | --- | --- |
| `current` | 6.5% [95% CI 2.2-17.5] (3/46) | 26.1% [95% CI 15.6-40.3] (12/46) | 50.0% [95% CI 36.1-63.9] (23/46) |
| `prior` | 27.5% [95% CI 16.1-42.8] (11/40) | 32.5% [95% CI 20.1-48.0] (13/40) | 95.0% [95% CI 83.5-98.6] (38/40) |
| `clarify` (auxiliary) | 100.0% [95% CI 91.2-100.0] (40/40) | 80.0% [95% CI 65.2-89.5] (32/40) | 15.0% [95% CI 7.1-29.1] (6/40) |
| `abstain` (auxiliary) | 52.5% [95% CI 37.5-67.1] (21/40) | 90.0% [95% CI 76.9-96.0] (36/40) | 60.0% [95% CI 44.6-73.7] (24/40) |

## Hedging behavior

- **Clarification rate**: 60.8% (95% CI 53.3%-67.9%)
- **Abstention rate**: 19.9% (95% CI 14.5%-26.6%)
- **Coverage** (1 - clarify - abstain): 19.3% (95% CI 14.0%-25.9%)
- _Coverage below 60% - model is hedging on a majority of trials. Compare against a less hedge-prone baseline._

## Code-judge disagreement by task

- task-001: 1 trial(s) with code/judge disagreement
- task-002: 0 trial(s) with code/judge disagreement
- task-003: 0 trial(s) with code/judge disagreement
- task-004: 0 trial(s) with code/judge disagreement
- task-005: 1 trial(s) with code/judge disagreement
- task-006: 2 trial(s) with code/judge disagreement
- task-007: 1 trial(s) with code/judge disagreement
- task-008: 1 trial(s) with code/judge disagreement
- task-009: 0 trial(s) with code/judge disagreement
- task-010: 2 trial(s) with code/judge disagreement
- task-011: 1 trial(s) with code/judge disagreement
- task-012: 0 trial(s) with code/judge disagreement
- task-013: 2 trial(s) with code/judge disagreement
- task-014: 2 trial(s) with code/judge disagreement
- task-015: 2 trial(s) with code/judge disagreement
- task-016: 0 trial(s) with code/judge disagreement
- task-017: 0 trial(s) with code/judge disagreement
- task-018: 1 trial(s) with code/judge disagreement
- task-019: 0 trial(s) with code/judge disagreement
- task-020: 0 trial(s) with code/judge disagreement
- task-021: 2 trial(s) with code/judge disagreement
- task-022: 0 trial(s) with code/judge disagreement
- task-023: 2 trial(s) with code/judge disagreement
- task-024: 0 trial(s) with code/judge disagreement
- task-025: 1 trial(s) with code/judge disagreement
- task-026: 2 trial(s) with code/judge disagreement
- task-027: 2 trial(s) with code/judge disagreement
- task-028: 1 trial(s) with code/judge disagreement
- task-029: 0 trial(s) with code/judge disagreement
- task-030: 1 trial(s) with code/judge disagreement
- task-031: 0 trial(s) with code/judge disagreement
- task-032: 1 trial(s) with code/judge disagreement
- task-033: 1 trial(s) with code/judge disagreement
- task-034: 0 trial(s) with code/judge disagreement
- task-035: 0 trial(s) with code/judge disagreement
- task-036: 0 trial(s) with code/judge disagreement
- task-037: 1 trial(s) with code/judge disagreement
- task-038: 0 trial(s) with code/judge disagreement
- task-039: 1 trial(s) with code/judge disagreement
- task-040: 0 trial(s) with code/judge disagreement
- task-041: 2 trial(s) with code/judge disagreement
- task-042: 1 trial(s) with code/judge disagreement
- task-043: 1 trial(s) with code/judge disagreement
- task-044: 0 trial(s) with code/judge disagreement
- task-045: 1 trial(s) with code/judge disagreement
- task-046: 0 trial(s) with code/judge disagreement
- task-047: 1 trial(s) with code/judge disagreement
- task-048: 0 trial(s) with code/judge disagreement
- task-049: 0 trial(s) with code/judge disagreement
- task-050: 2 trial(s) with code/judge disagreement
- task-051: 0 trial(s) with code/judge disagreement
- task-052: 0 trial(s) with code/judge disagreement
- task-053: 1 trial(s) with code/judge disagreement
- task-054: 1 trial(s) with code/judge disagreement
- task-055: 0 trial(s) with code/judge disagreement
- task-056: 0 trial(s) with code/judge disagreement
- task-057: 0 trial(s) with code/judge disagreement
- task-058: 1 trial(s) with code/judge disagreement
- task-059: 0 trial(s) with code/judge disagreement
- task-060: 0 trial(s) with code/judge disagreement
- task-061: 0 trial(s) with code/judge disagreement
- task-062: 1 trial(s) with code/judge disagreement
- task-063: 1 trial(s) with code/judge disagreement
- task-064: 1 trial(s) with code/judge disagreement
- task-065: 1 trial(s) with code/judge disagreement
- task-066: 0 trial(s) with code/judge disagreement
- task-067: 0 trial(s) with code/judge disagreement
- task-068: 0 trial(s) with code/judge disagreement
- task-069: 0 trial(s) with code/judge disagreement
- task-070: 0 trial(s) with code/judge disagreement
- task-071: 1 trial(s) with code/judge disagreement
- task-072: 1 trial(s) with code/judge disagreement
- task-073: 0 trial(s) with code/judge disagreement
- task-074: 2 trial(s) with code/judge disagreement
- task-075: 0 trial(s) with code/judge disagreement
- task-076: 0 trial(s) with code/judge disagreement
- task-077: 0 trial(s) with code/judge disagreement
- task-078: 0 trial(s) with code/judge disagreement
- task-079: 1 trial(s) with code/judge disagreement
- task-080: 2 trial(s) with code/judge disagreement
- task-081: 0 trial(s) with code/judge disagreement
- task-082: 0 trial(s) with code/judge disagreement
- task-083: 1 trial(s) with code/judge disagreement
- task-084: 0 trial(s) with code/judge disagreement
- task-085: 0 trial(s) with code/judge disagreement
- task-086: 0 trial(s) with code/judge disagreement
- task-087: 0 trial(s) with code/judge disagreement
- task-088: 3 trial(s) with code/judge disagreement
- task-089: 1 trial(s) with code/judge disagreement
- task-090: 1 trial(s) with code/judge disagreement
- task-091: 1 trial(s) with code/judge disagreement
- task-092: 0 trial(s) with code/judge disagreement
- task-093: 0 trial(s) with code/judge disagreement
- task-094: 1 trial(s) with code/judge disagreement
- task-095: 0 trial(s) with code/judge disagreement
- task-096: 0 trial(s) with code/judge disagreement
- task-097: 1 trial(s) with code/judge disagreement
- task-098: 2 trial(s) with code/judge disagreement
- task-099: 0 trial(s) with code/judge disagreement
- task-100: 0 trial(s) with code/judge disagreement
- task-101: 0 trial(s) with code/judge disagreement
- task-102: 0 trial(s) with code/judge disagreement
- task-103: 0 trial(s) with code/judge disagreement
- task-104: 0 trial(s) with code/judge disagreement
- task-105: 0 trial(s) with code/judge disagreement
- task-106: 0 trial(s) with code/judge disagreement
- task-107: 1 trial(s) with code/judge disagreement
- task-108: 1 trial(s) with code/judge disagreement
- task-109: 0 trial(s) with code/judge disagreement
- task-110: 0 trial(s) with code/judge disagreement
- task-111: 1 trial(s) with code/judge disagreement
- task-112: 0 trial(s) with code/judge disagreement
- task-113: 2 trial(s) with code/judge disagreement
- task-114: 0 trial(s) with code/judge disagreement
- task-115: 0 trial(s) with code/judge disagreement
- task-116: 2 trial(s) with code/judge disagreement
- task-117: 2 trial(s) with code/judge disagreement
- task-118: 0 trial(s) with code/judge disagreement
- task-119: 0 trial(s) with code/judge disagreement
- task-120: 0 trial(s) with code/judge disagreement
- task-121: 2 trial(s) with code/judge disagreement
- task-122: 0 trial(s) with code/judge disagreement
- task-123: 0 trial(s) with code/judge disagreement
- task-124: 0 trial(s) with code/judge disagreement
- task-125: 1 trial(s) with code/judge disagreement
- task-126: 0 trial(s) with code/judge disagreement
- task-127: 0 trial(s) with code/judge disagreement
- task-128: 2 trial(s) with code/judge disagreement
- task-129: 0 trial(s) with code/judge disagreement
- task-130: 0 trial(s) with code/judge disagreement
- task-131: 0 trial(s) with code/judge disagreement
- task-132: 1 trial(s) with code/judge disagreement
- task-133: 0 trial(s) with code/judge disagreement
- task-134: 1 trial(s) with code/judge disagreement
- task-135: 1 trial(s) with code/judge disagreement
- task-136: 0 trial(s) with code/judge disagreement
- task-137: 0 trial(s) with code/judge disagreement
- task-138: 1 trial(s) with code/judge disagreement
- task-139: 0 trial(s) with code/judge disagreement
- task-140: 0 trial(s) with code/judge disagreement
- task-141: 0 trial(s) with code/judge disagreement
- task-142: 0 trial(s) with code/judge disagreement
- task-143: 0 trial(s) with code/judge disagreement
- task-144: 0 trial(s) with code/judge disagreement
- task-145: 0 trial(s) with code/judge disagreement
- task-146: 0 trial(s) with code/judge disagreement
- task-147: 0 trial(s) with code/judge disagreement
- task-148: 0 trial(s) with code/judge disagreement
- task-149: 0 trial(s) with code/judge disagreement
- task-150: 0 trial(s) with code/judge disagreement
- task-151: 0 trial(s) with code/judge disagreement
- task-152: 0 trial(s) with code/judge disagreement
- task-153: 0 trial(s) with code/judge disagreement
- task-154: 1 trial(s) with code/judge disagreement
- task-155: 0 trial(s) with code/judge disagreement
- task-156: 0 trial(s) with code/judge disagreement
- task-157: 0 trial(s) with code/judge disagreement
- task-158: 0 trial(s) with code/judge disagreement
- task-159: 0 trial(s) with code/judge disagreement
- task-160: 0 trial(s) with code/judge disagreement
- task-161: 0 trial(s) with code/judge disagreement
- task-162: 1 trial(s) with code/judge disagreement
- task-163: 0 trial(s) with code/judge disagreement
- task-164: 0 trial(s) with code/judge disagreement
- task-165: 0 trial(s) with code/judge disagreement
- task-166: 0 trial(s) with code/judge disagreement

## Inter-judge agreement (cross-LLM)

_No ranking-judge labels in this run. To enable cross-LLM inter-judge agreement, pass `--ranking-judge-family` to the runner so every trial is also labeled by a fixed second judge._

## Task-by-condition matrix

| Task | Target context | baseline | context_selection_instruction | pre_answer_context_scaffold |
| --- | --- | --- | --- | --- |
| task-001 | `current` | fail | fail | pass |
| task-002 | `current` | fail | fail | fail |
| task-003 | `current` | fail | fail | fail |
| task-004 | `prior` | fail | pass | pass |
| task-005 | `current` | fail | fail | pass |
| task-006 | `current` | fail | fail | fail |
| task-007 | `current` | fail | fail | fail |
| task-008 | `current` | pass | pass | fail |
| task-009 | `clarify` | pass | pass | fail |
| task-010 | `current` | fail | fail | pass |
| task-011 | `current` | fail | fail | fail |
| task-012 | `prior` | fail | fail | pass |
| task-013 | `current` | fail | pass | pass |
| task-014 | `current` | fail | fail | pass |
| task-015 | `current` | fail | fail | pass |
| task-016 | `prior` | pass | fail | pass |
| task-017 | `current` | fail | fail | pass |
| task-018 | `current` | fail | fail | pass |
| task-019 | `prior` | pass | fail | pass |
| task-020 | `current` | fail | pass | pass |
| task-021 | `current` | fail | fail | pass |
| task-022 | `current` | fail | fail | pass |
| task-023 | `current` | fail | fail | fail |
| task-024 | `prior` | fail | fail | pass |
| task-025 | `current` | fail | pass | pass |
| task-026 | `current` | fail | fail | fail |
| task-027 | `current` | pass | pass | pass |
| task-028 | `current` | fail | fail | pass |
| task-029 | `current` | fail | fail | pass |
| task-030 | `current` | fail | fail | fail |
| task-031 | `prior` | fail | fail | pass |
| task-032 | `clarify` | pass | pass | fail |
| task-033 | `current` | fail | fail | fail |
| task-034 | `current` | fail | fail | fail |
| task-035 | `current` | fail | pass | fail |
| task-036 | `current` | fail | pass | fail |
| task-037 | `prior` | fail | pass | pass |
| task-038 | `prior` | pass | fail | pass |
| task-039 | `abstain` | fail | pass | pass |
| task-040 | `current` | pass | fail | pass |
| task-041 | `clarify` | pass | pass | fail |
| task-042 | `abstain` | fail | pass | fail |
| task-043 | `current` | fail | fail | pass |
| task-044 | `current` | fail | pass | fail |
| task-045 | `current` | fail | fail | fail |
| task-046 | `prior` | fail | fail | pass |
| task-047 | `prior` | fail | fail | fail |
| task-048 | `prior` | fail | pass | pass |
| task-049 | `prior` | fail | pass | pass |
| task-050 | `current` | fail | pass | pass |
| task-051 | `current` | fail | fail | fail |
| task-052 | `current` | fail | fail | pass |
| task-053 | `current` | fail | fail | pass |
| task-054 | `current` | fail | fail | pass |
| task-055 | `current` | fail | fail | fail |
| task-056 | `current` | fail | pass | fail |
| task-057 | `current` | fail | fail | pass |
| task-058 | `current` | fail | fail | fail |
| task-059 | `current` | fail | pass | fail |
| task-060 | `current` | fail | pass | pass |
| task-061 | `current` | fail | fail | fail |
| task-062 | `current` | fail | fail | fail |
| task-063 | `current` | fail | fail | fail |
| task-064 | `prior` | fail | pass | pass |
| task-065 | `prior` | fail | fail | pass |
| task-066 | `prior` | pass | pass | pass |
| task-067 | `prior` | pass | pass | pass |
| task-068 | `clarify` | pass | pass | pass |
| task-069 | `clarify` | pass | pass | fail |
| task-070 | `abstain` | fail | pass | pass |
| task-071 | `prior` | pass | fail | pass |
| task-072 | `clarify` | pass | pass | fail |
| task-073 | `abstain` | fail | fail | pass |
| task-074 | `prior` | fail | pass | pass |
| task-075 | `prior` | fail | fail | pass |
| task-076 | `prior` | pass | pass | pass |
| task-077 | `clarify` | pass | pass | fail |
| task-078 | `clarify` | pass | pass | fail |
| task-079 | `clarify` | pass | pass | fail |
| task-080 | `clarify` | pass | pass | fail |
| task-081 | `clarify` | pass | fail | fail |
| task-082 | `abstain` | fail | pass | fail |
| task-083 | `abstain` | fail | fail | pass |
| task-084 | `prior` | fail | pass | pass |
| task-085 | `prior` | pass | fail | fail |
| task-086 | `clarify` | pass | pass | fail |
| task-087 | `clarify` | pass | pass | fail |
| task-088 | `clarify` | pass | fail | pass |
| task-089 | `clarify` | pass | pass | fail |
| task-090 | `clarify` | pass | pass | pass |
| task-091 | `clarify` | pass | pass | fail |
| task-092 | `clarify` | pass | pass | pass |
| task-093 | `clarify` | pass | pass | fail |
| task-094 | `clarify` | pass | pass | fail |
| task-095 | `abstain` | fail | fail | fail |
| task-096 | `prior` | fail | fail | pass |
| task-097 | `prior` | fail | fail | pass |
| task-098 | `prior` | fail | fail | pass |
| task-099 | `clarify` | pass | pass | fail |
| task-100 | `clarify` | pass | pass | fail |
| task-101 | `clarify` | pass | pass | fail |
| task-102 | `clarify` | pass | pass | fail |
| task-103 | `clarify` | pass | pass | fail |
| task-104 | `abstain` | fail | pass | pass |
| task-105 | `abstain` | fail | pass | fail |
| task-106 | `abstain` | fail | pass | pass |
| task-107 | `abstain` | fail | pass | pass |
| task-108 | `prior` | fail | fail | pass |
| task-109 | `prior` | fail | pass | pass |
| task-110 | `prior` | fail | fail | pass |
| task-111 | `clarify` | pass | pass | fail |
| task-112 | `clarify` | pass | fail | fail |
| task-113 | `clarify` | pass | pass | fail |
| task-114 | `clarify` | pass | fail | fail |
| task-115 | `clarify` | pass | pass | fail |
| task-116 | `clarify` | pass | pass | fail |
| task-117 | `clarify` | pass | pass | fail |
| task-118 | `abstain` | fail | pass | pass |
| task-119 | `abstain` | fail | pass | fail |
| task-120 | `abstain` | pass | pass | fail |
| task-121 | `prior` | fail | fail | pass |
| task-122 | `prior` | fail | fail | pass |
| task-123 | `prior` | pass | pass | pass |
| task-124 | `prior` | fail | fail | pass |
| task-125 | `clarify` | pass | fail | fail |
| task-126 | `clarify` | pass | fail | fail |
| task-127 | `clarify` | pass | pass | fail |
| task-128 | `clarify` | pass | pass | pass |
| task-129 | `abstain` | pass | pass | pass |
| task-130 | `abstain` | pass | pass | fail |
| task-131 | `abstain` | fail | pass | fail |
| task-132 | `abstain` | pass | pass | pass |
| task-133 | `abstain` | pass | pass | pass |
| task-134 | `abstain` | fail | pass | pass |
| task-135 | `abstain` | fail | pass | pass |
| task-136 | `prior` | fail | fail | pass |
| task-137 | `prior` | fail | fail | pass |
| task-138 | `prior` | pass | fail | pass |
| task-139 | `clarify` | pass | pass | fail |
| task-140 | `clarify` | pass | fail | fail |
| task-141 | `clarify` | pass | fail | fail |
| task-142 | `clarify` | pass | pass | pass |
| task-143 | `abstain` | fail | fail | pass |
| task-144 | `abstain` | pass | pass | fail |
| task-145 | `abstain` | pass | pass | pass |
| task-146 | `abstain` | pass | pass | pass |
| task-147 | `abstain` | pass | pass | pass |
| task-148 | `abstain` | pass | pass | pass |
| task-149 | `abstain` | pass | pass | fail |
| task-150 | `abstain` | pass | pass | fail |
| task-151 | `prior` | fail | fail | pass |
| task-152 | `prior` | fail | pass | pass |
| task-153 | `prior` | fail | fail | pass |
| task-154 | `prior` | pass | fail | pass |
| task-155 | `prior` | fail | fail | pass |
| task-156 | `abstain` | pass | pass | fail |
| task-157 | `abstain` | pass | pass | pass |
| task-158 | `abstain` | pass | pass | pass |
| task-159 | `abstain` | pass | pass | fail |
| task-160 | `abstain` | pass | pass | fail |
| task-161 | `abstain` | fail | pass | pass |
| task-162 | `abstain` | fail | pass | fail |
| task-163 | `abstain` | pass | pass | fail |
| task-164 | `abstain` | pass | pass | pass |
| task-165 | `abstain` | pass | pass | pass |
| task-166 | `abstain` | pass | pass | pass |

## Reproducibility manifest

```json
{
  "benchmark_version": "0.1.0a0",
  "tasks_sha256": "bd3611d20996a3b3d6a6a1d9c9833f36b5a287bb4f746d26cec2537777b157d1",
  "prompt_conditions_sha256": "292f7dc0631c956c850ba868106c395f1d0865589ebd123df1a5471095b77bce",
  "candidate_model": "gemini/gemini-2.5-flash-lite",
  "judge_model": "gemini/gemini-2.5-flash-lite",
  "judge_family": "gemini",
  "trials": 1,
  "temperature": 0.0,
  "ranking_condition": "baseline",
  "timestamp_utc": "2026-05-06T19:38:03+00:00",
  "runner_git_commit": "b31185f820fbc6bbb36dc11231ad87af9514a062",
  "random_seed": null,
  "task_set": "main",
  "camera_injection": false,
  "judge_prompt_sha256": "b2289bcc4da706408132b89a4dbe1cdcd83423921779f69adb6eab8e2a03bec5",
  "judge_family_resolution": "explicit",
  "ranking_judge_model": null,
  "ranking_judge_family": null,
  "enable_repair": false,
  "manifest_warnings": []
}
```
