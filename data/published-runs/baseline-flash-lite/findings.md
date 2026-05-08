# Wearable Assistant Context Bench: Findings

**Benchmark:** situated context-tracking benchmark for AI wearable assistants used actively for advice or coaching

## Benchmark summary

- **Benchmark**: situated context-tracking benchmark for AI wearable assistants used actively for advice or coaching
- **Default comparison condition**: `baseline`
- **Primary score** - `mean(current_recall, prior_recall)` (class recall, not overall accuracy): **54.1% (95% CI 44.9%-63.4%)**
- **Bootstrap 95% CI on primary score**: 54.1% (95% CI 44.9%-63.3%) (5000 percentile bootstrap iterations)
- **How to read this run**: compare candidate models on the `baseline` score below; treat the other conditions as diagnostic sensitivity checks. CIs are 95% Wilson per class and 95% normal-approximation on the mean recall (with a bootstrap second opinion).
- **Per-class recall under `baseline`** (TP / (TP + FN)):
    - `current_recall`: 78.3% (95% CI 64.4%-87.7%)
    - `prior_recall`: 30.0% (95% CI 18.1%-45.4%)

Condition sensitivity (mean per-class recall):

| Condition | Mean recall (95% CI) |
| --- | --- |
| baseline (default) | 54.1% (95% CI 44.9%-63.4%) |
| context_selection_instruction | 72.2% (95% CI 63.2%-81.3%) |
| pre_answer_context_scaffold | 91.7% (95% CI 85.9%-97.6%) |

## Per task-set recall

| Task set | Mean recall (95% CI) |
| --- | --- |
| `main` | 54.1% (95% CI 44.9%-63.4%) |

## Per shift-type recall

| Shift type | Pass rate (95% CI) |
| --- | --- |
| `absent_referent` | 52.4% (95% CI 32.4%-71.7%) |
| `cross_session_reference` | 80.0% (95% CI 58.4%-91.9%) |
| `location` | 31.8% (95% CI 16.4%-52.7%) |
| `object_in_hand` | 57.1% (95% CI 36.5%-75.5%) |
| `object_in_view` | 38.1% (95% CI 20.8%-59.1%) |
| `object_state` | 50.0% (95% CI 30.7%-69.3%) |
| `screen_content` | 61.9% (95% CI 40.9%-79.2%) |
| `sequential_task` | 61.1% (95% CI 38.6%-79.7%) |

## Per-class pass rate by condition

| Class | baseline | context_selection_instruction | pre_answer_context_scaffold |
| --- | --- | --- | --- |
| `current` | 78.3% [95% CI 64.4-87.7] (36/46) | 87.0% [95% CI 74.3-93.9] (40/46) | 93.5% [95% CI 82.5-97.8] (43/46) |
| `prior` | 30.0% [95% CI 18.1-45.4] (12/40) | 57.5% [95% CI 42.2-71.5] (23/40) | 90.0% [95% CI 76.9-96.0] (36/40) |
| `clarify` (auxiliary) | 35.0% [95% CI 22.1-50.5] (14/40) | 40.0% [95% CI 26.3-55.4] (16/40) | 5.0% [95% CI 1.4-16.5] (2/40) |
| `abstain` (auxiliary) | 67.5% [95% CI 52.0-79.9] (27/40) | 77.5% [95% CI 62.5-87.7] (31/40) | 32.5% [95% CI 20.1-48.0] (13/40) |

## Hedging behavior

- **Clarification rate**: 19.3% (95% CI 14.0%-25.9%)
- **Abstention rate**: 18.1% (95% CI 13.0%-24.6%)
- **Coverage** (1 - clarify - abstain): 62.7% (95% CI 55.1%-69.6%)

## Code-judge disagreement by task

- task-001: 0 trial(s) with code/judge disagreement
- task-002: 0 trial(s) with code/judge disagreement
- task-003: 1 trial(s) with code/judge disagreement
- task-004: 0 trial(s) with code/judge disagreement
- task-005: 0 trial(s) with code/judge disagreement
- task-006: 1 trial(s) with code/judge disagreement
- task-007: 0 trial(s) with code/judge disagreement
- task-008: 1 trial(s) with code/judge disagreement
- task-009: 1 trial(s) with code/judge disagreement
- task-010: 0 trial(s) with code/judge disagreement
- task-011: 0 trial(s) with code/judge disagreement
- task-012: 1 trial(s) with code/judge disagreement
- task-013: 0 trial(s) with code/judge disagreement
- task-014: 0 trial(s) with code/judge disagreement
- task-015: 0 trial(s) with code/judge disagreement
- task-016: 0 trial(s) with code/judge disagreement
- task-017: 0 trial(s) with code/judge disagreement
- task-018: 0 trial(s) with code/judge disagreement
- task-019: 0 trial(s) with code/judge disagreement
- task-020: 0 trial(s) with code/judge disagreement
- task-021: 0 trial(s) with code/judge disagreement
- task-022: 0 trial(s) with code/judge disagreement
- task-023: 0 trial(s) with code/judge disagreement
- task-024: 0 trial(s) with code/judge disagreement
- task-025: 0 trial(s) with code/judge disagreement
- task-026: 0 trial(s) with code/judge disagreement
- task-027: 0 trial(s) with code/judge disagreement
- task-028: 1 trial(s) with code/judge disagreement
- task-029: 1 trial(s) with code/judge disagreement
- task-030: 0 trial(s) with code/judge disagreement
- task-031: 2 trial(s) with code/judge disagreement
- task-032: 0 trial(s) with code/judge disagreement
- task-033: 0 trial(s) with code/judge disagreement
- task-034: 0 trial(s) with code/judge disagreement
- task-035: 1 trial(s) with code/judge disagreement
- task-036: 0 trial(s) with code/judge disagreement
- task-037: 1 trial(s) with code/judge disagreement
- task-038: 1 trial(s) with code/judge disagreement
- task-039: 0 trial(s) with code/judge disagreement
- task-040: 0 trial(s) with code/judge disagreement
- task-041: 3 trial(s) with code/judge disagreement
- task-042: 1 trial(s) with code/judge disagreement
- task-043: 1 trial(s) with code/judge disagreement
- task-044: 0 trial(s) with code/judge disagreement
- task-045: 0 trial(s) with code/judge disagreement
- task-046: 3 trial(s) with code/judge disagreement
- task-047: 0 trial(s) with code/judge disagreement
- task-048: 0 trial(s) with code/judge disagreement
- task-049: 0 trial(s) with code/judge disagreement
- task-050: 0 trial(s) with code/judge disagreement
- task-051: 0 trial(s) with code/judge disagreement
- task-052: 0 trial(s) with code/judge disagreement
- task-053: 0 trial(s) with code/judge disagreement
- task-054: 0 trial(s) with code/judge disagreement
- task-055: 0 trial(s) with code/judge disagreement
- task-056: 0 trial(s) with code/judge disagreement
- task-057: 2 trial(s) with code/judge disagreement
- task-058: 0 trial(s) with code/judge disagreement
- task-059: 0 trial(s) with code/judge disagreement
- task-060: 0 trial(s) with code/judge disagreement
- task-061: 1 trial(s) with code/judge disagreement
- task-062: 0 trial(s) with code/judge disagreement
- task-063: 1 trial(s) with code/judge disagreement
- task-064: 0 trial(s) with code/judge disagreement
- task-065: 2 trial(s) with code/judge disagreement
- task-066: 0 trial(s) with code/judge disagreement
- task-067: 2 trial(s) with code/judge disagreement
- task-068: 0 trial(s) with code/judge disagreement
- task-069: 0 trial(s) with code/judge disagreement
- task-070: 0 trial(s) with code/judge disagreement
- task-071: 0 trial(s) with code/judge disagreement
- task-072: 0 trial(s) with code/judge disagreement
- task-073: 2 trial(s) with code/judge disagreement
- task-074: 0 trial(s) with code/judge disagreement
- task-075: 0 trial(s) with code/judge disagreement
- task-076: 0 trial(s) with code/judge disagreement
- task-077: 0 trial(s) with code/judge disagreement
- task-078: 0 trial(s) with code/judge disagreement
- task-079: 0 trial(s) with code/judge disagreement
- task-080: 0 trial(s) with code/judge disagreement
- task-081: 0 trial(s) with code/judge disagreement
- task-082: 0 trial(s) with code/judge disagreement
- task-083: 0 trial(s) with code/judge disagreement
- task-084: 0 trial(s) with code/judge disagreement
- task-085: 1 trial(s) with code/judge disagreement
- task-086: 0 trial(s) with code/judge disagreement
- task-087: 0 trial(s) with code/judge disagreement
- task-088: 1 trial(s) with code/judge disagreement
- task-089: 0 trial(s) with code/judge disagreement
- task-090: 0 trial(s) with code/judge disagreement
- task-091: 0 trial(s) with code/judge disagreement
- task-092: 1 trial(s) with code/judge disagreement
- task-093: 0 trial(s) with code/judge disagreement
- task-094: 0 trial(s) with code/judge disagreement
- task-095: 0 trial(s) with code/judge disagreement
- task-096: 1 trial(s) with code/judge disagreement
- task-097: 1 trial(s) with code/judge disagreement
- task-098: 2 trial(s) with code/judge disagreement
- task-099: 1 trial(s) with code/judge disagreement
- task-100: 2 trial(s) with code/judge disagreement
- task-101: 0 trial(s) with code/judge disagreement
- task-102: 0 trial(s) with code/judge disagreement
- task-103: 2 trial(s) with code/judge disagreement
- task-104: 0 trial(s) with code/judge disagreement
- task-105: 1 trial(s) with code/judge disagreement
- task-106: 1 trial(s) with code/judge disagreement
- task-107: 0 trial(s) with code/judge disagreement
- task-108: 0 trial(s) with code/judge disagreement
- task-109: 2 trial(s) with code/judge disagreement
- task-110: 0 trial(s) with code/judge disagreement
- task-111: 1 trial(s) with code/judge disagreement
- task-112: 0 trial(s) with code/judge disagreement
- task-113: 0 trial(s) with code/judge disagreement
- task-114: 0 trial(s) with code/judge disagreement
- task-115: 0 trial(s) with code/judge disagreement
- task-116: 0 trial(s) with code/judge disagreement
- task-117: 0 trial(s) with code/judge disagreement
- task-118: 0 trial(s) with code/judge disagreement
- task-119: 3 trial(s) with code/judge disagreement
- task-120: 1 trial(s) with code/judge disagreement
- task-121: 2 trial(s) with code/judge disagreement
- task-122: 1 trial(s) with code/judge disagreement
- task-123: 1 trial(s) with code/judge disagreement
- task-124: 1 trial(s) with code/judge disagreement
- task-125: 2 trial(s) with code/judge disagreement
- task-126: 0 trial(s) with code/judge disagreement
- task-127: 1 trial(s) with code/judge disagreement
- task-128: 1 trial(s) with code/judge disagreement
- task-129: 0 trial(s) with code/judge disagreement
- task-130: 0 trial(s) with code/judge disagreement
- task-131: 1 trial(s) with code/judge disagreement
- task-132: 0 trial(s) with code/judge disagreement
- task-133: 2 trial(s) with code/judge disagreement
- task-134: 2 trial(s) with code/judge disagreement
- task-135: 0 trial(s) with code/judge disagreement
- task-136: 0 trial(s) with code/judge disagreement
- task-137: 0 trial(s) with code/judge disagreement
- task-138: 1 trial(s) with code/judge disagreement
- task-139: 0 trial(s) with code/judge disagreement
- task-140: 0 trial(s) with code/judge disagreement
- task-141: 0 trial(s) with code/judge disagreement
- task-142: 0 trial(s) with code/judge disagreement
- task-143: 0 trial(s) with code/judge disagreement
- task-144: 1 trial(s) with code/judge disagreement
- task-145: 0 trial(s) with code/judge disagreement
- task-146: 0 trial(s) with code/judge disagreement
- task-147: 0 trial(s) with code/judge disagreement
- task-148: 1 trial(s) with code/judge disagreement
- task-149: 0 trial(s) with code/judge disagreement
- task-150: 0 trial(s) with code/judge disagreement
- task-151: 0 trial(s) with code/judge disagreement
- task-152: 0 trial(s) with code/judge disagreement
- task-153: 0 trial(s) with code/judge disagreement
- task-154: 0 trial(s) with code/judge disagreement
- task-155: 0 trial(s) with code/judge disagreement
- task-156: 0 trial(s) with code/judge disagreement
- task-157: 1 trial(s) with code/judge disagreement
- task-158: 1 trial(s) with code/judge disagreement
- task-159: 0 trial(s) with code/judge disagreement
- task-160: 0 trial(s) with code/judge disagreement
- task-161: 0 trial(s) with code/judge disagreement
- task-162: 0 trial(s) with code/judge disagreement
- task-163: 0 trial(s) with code/judge disagreement
- task-164: 1 trial(s) with code/judge disagreement
- task-165: 1 trial(s) with code/judge disagreement
- task-166: 0 trial(s) with code/judge disagreement

## Inter-judge agreement (cross-LLM)

_No ranking-judge labels in this run. To enable cross-LLM inter-judge agreement, pass `--ranking-judge-family` to the runner so every trial is also labeled by a fixed second judge._

## Task-by-condition matrix

| Task | Target context | baseline | context_selection_instruction | pre_answer_context_scaffold |
| --- | --- | --- | --- | --- |
| task-001 | `current` | pass | pass | pass |
| task-002 | `current` | pass | fail | pass |
| task-003 | `current` | fail | pass | pass |
| task-004 | `prior` | fail | pass | pass |
| task-005 | `current` | pass | pass | pass |
| task-006 | `current` | pass | pass | pass |
| task-007 | `current` | pass | pass | pass |
| task-008 | `current` | pass | fail | pass |
| task-009 | `clarify` | pass | pass | fail |
| task-010 | `current` | pass | pass | fail |
| task-011 | `current` | pass | pass | pass |
| task-012 | `prior` | fail | fail | pass |
| task-013 | `current` | pass | pass | pass |
| task-014 | `current` | pass | pass | pass |
| task-015 | `current` | pass | pass | pass |
| task-016 | `prior` | fail | fail | pass |
| task-017 | `current` | pass | pass | pass |
| task-018 | `current` | pass | pass | pass |
| task-019 | `prior` | fail | pass | pass |
| task-020 | `current` | pass | pass | pass |
| task-021 | `current` | pass | pass | pass |
| task-022 | `current` | pass | pass | pass |
| task-023 | `current` | pass | pass | pass |
| task-024 | `prior` | fail | pass | pass |
| task-025 | `current` | pass | pass | pass |
| task-026 | `current` | pass | pass | pass |
| task-027 | `current` | pass | pass | pass |
| task-028 | `current` | fail | fail | pass |
| task-029 | `current` | fail | pass | pass |
| task-030 | `current` | pass | pass | pass |
| task-031 | `prior` | pass | fail | pass |
| task-032 | `clarify` | fail | fail | fail |
| task-033 | `current` | pass | pass | pass |
| task-034 | `current` | fail | pass | pass |
| task-035 | `current` | pass | pass | pass |
| task-036 | `current` | pass | pass | pass |
| task-037 | `prior` | fail | fail | pass |
| task-038 | `prior` | fail | pass | pass |
| task-039 | `abstain` | fail | pass | fail |
| task-040 | `current` | pass | pass | pass |
| task-041 | `clarify` | pass | fail | fail |
| task-042 | `abstain` | fail | pass | fail |
| task-043 | `current` | fail | fail | pass |
| task-044 | `current` | pass | pass | pass |
| task-045 | `current` | pass | fail | pass |
| task-046 | `prior` | fail | pass | pass |
| task-047 | `prior` | pass | pass | pass |
| task-048 | `prior` | pass | pass | pass |
| task-049 | `prior` | pass | fail | pass |
| task-050 | `current` | pass | pass | pass |
| task-051 | `current` | pass | pass | pass |
| task-052 | `current` | fail | pass | pass |
| task-053 | `current` | pass | pass | pass |
| task-054 | `current` | pass | pass | pass |
| task-055 | `current` | pass | pass | pass |
| task-056 | `current` | pass | pass | pass |
| task-057 | `current` | pass | pass | fail |
| task-058 | `current` | fail | pass | pass |
| task-059 | `current` | pass | fail | pass |
| task-060 | `current` | pass | pass | pass |
| task-061 | `current` | fail | pass | fail |
| task-062 | `current` | fail | pass | pass |
| task-063 | `current` | fail | pass | pass |
| task-064 | `prior` | fail | pass | fail |
| task-065 | `prior` | fail | fail | pass |
| task-066 | `prior` | pass | pass | pass |
| task-067 | `prior` | pass | pass | pass |
| task-068 | `clarify` | fail | fail | fail |
| task-069 | `clarify` | fail | fail | fail |
| task-070 | `abstain` | pass | pass | fail |
| task-071 | `prior` | fail | fail | fail |
| task-072 | `clarify` | pass | pass | fail |
| task-073 | `abstain` | fail | pass | fail |
| task-074 | `prior` | fail | pass | pass |
| task-075 | `prior` | fail | fail | pass |
| task-076 | `prior` | fail | pass | pass |
| task-077 | `clarify` | fail | fail | fail |
| task-078 | `clarify` | fail | fail | fail |
| task-079 | `clarify` | fail | fail | fail |
| task-080 | `clarify` | pass | fail | fail |
| task-081 | `clarify` | fail | fail | fail |
| task-082 | `abstain` | fail | fail | fail |
| task-083 | `abstain` | fail | fail | fail |
| task-084 | `prior` | fail | fail | pass |
| task-085 | `prior` | fail | fail | pass |
| task-086 | `clarify` | fail | fail | fail |
| task-087 | `clarify` | fail | pass | fail |
| task-088 | `clarify` | fail | pass | fail |
| task-089 | `clarify` | fail | pass | fail |
| task-090 | `clarify` | pass | pass | pass |
| task-091 | `clarify` | fail | pass | pass |
| task-092 | `clarify` | pass | pass | fail |
| task-093 | `clarify` | fail | pass | fail |
| task-094 | `clarify` | fail | pass | fail |
| task-095 | `abstain` | pass | pass | pass |
| task-096 | `prior` | fail | fail | pass |
| task-097 | `prior` | fail | pass | pass |
| task-098 | `prior` | pass | fail | pass |
| task-099 | `clarify` | fail | fail | fail |
| task-100 | `clarify` | fail | pass | fail |
| task-101 | `clarify` | pass | pass | fail |
| task-102 | `clarify` | fail | fail | fail |
| task-103 | `clarify` | fail | pass | fail |
| task-104 | `abstain` | pass | pass | pass |
| task-105 | `abstain` | fail | pass | fail |
| task-106 | `abstain` | pass | fail | pass |
| task-107 | `abstain` | pass | pass | pass |
| task-108 | `prior` | fail | fail | pass |
| task-109 | `prior` | fail | pass | pass |
| task-110 | `prior` | fail | fail | fail |
| task-111 | `clarify` | fail | fail | fail |
| task-112 | `clarify` | fail | fail | fail |
| task-113 | `clarify` | fail | fail | fail |
| task-114 | `clarify` | fail | fail | fail |
| task-115 | `clarify` | fail | fail | fail |
| task-116 | `clarify` | pass | fail | fail |
| task-117 | `clarify` | fail | fail | fail |
| task-118 | `abstain` | pass | pass | fail |
| task-119 | `abstain` | fail | fail | fail |
| task-120 | `abstain` | pass | pass | fail |
| task-121 | `prior` | fail | fail | pass |
| task-122 | `prior` | fail | pass | pass |
| task-123 | `prior` | fail | pass | pass |
| task-124 | `prior` | fail | pass | pass |
| task-125 | `clarify` | pass | fail | fail |
| task-126 | `clarify` | pass | pass | fail |
| task-127 | `clarify` | pass | fail | fail |
| task-128 | `clarify` | fail | pass | fail |
| task-129 | `abstain` | pass | fail | fail |
| task-130 | `abstain` | pass | pass | fail |
| task-131 | `abstain` | fail | pass | fail |
| task-132 | `abstain` | pass | pass | pass |
| task-133 | `abstain` | fail | fail | fail |
| task-134 | `abstain` | pass | fail | fail |
| task-135 | `abstain` | pass | pass | fail |
| task-136 | `prior` | pass | pass | pass |
| task-137 | `prior` | fail | fail | fail |
| task-138 | `prior` | fail | fail | pass |
| task-139 | `clarify` | pass | pass | fail |
| task-140 | `clarify` | pass | fail | fail |
| task-141 | `clarify` | fail | fail | fail |
| task-142 | `clarify` | pass | fail | fail |
| task-143 | `abstain` | pass | pass | pass |
| task-144 | `abstain` | pass | fail | fail |
| task-145 | `abstain` | pass | pass | pass |
| task-146 | `abstain` | fail | pass | fail |
| task-147 | `abstain` | pass | pass | pass |
| task-148 | `abstain` | pass | pass | fail |
| task-149 | `abstain` | pass | pass | fail |
| task-150 | `abstain` | pass | pass | pass |
| task-151 | `prior` | pass | pass | pass |
| task-152 | `prior` | pass | pass | pass |
| task-153 | `prior` | fail | pass | pass |
| task-154 | `prior` | pass | pass | pass |
| task-155 | `prior` | pass | pass | pass |
| task-156 | `abstain` | pass | pass | fail |
| task-157 | `abstain` | pass | pass | fail |
| task-158 | `abstain` | pass | pass | fail |
| task-159 | `abstain` | pass | pass | fail |
| task-160 | `abstain` | fail | pass | fail |
| task-161 | `abstain` | fail | pass | pass |
| task-162 | `abstain` | pass | pass | fail |
| task-163 | `abstain` | pass | pass | pass |
| task-164 | `abstain` | pass | fail | pass |
| task-165 | `abstain` | pass | pass | fail |
| task-166 | `abstain` | fail | pass | pass |

## Reproducibility manifest

```json
{
  "benchmark_version": "0.1.0a0",
  "tasks_sha256": "1b2fe36f2d19971a4f66e9f47462bac75926b38656045ce9afb604c1fc7b1de9",
  "prompt_conditions_sha256": "292f7dc0631c956c850ba868106c395f1d0865589ebd123df1a5471095b77bce",
  "candidate_model": "gemini/gemini-2.5-flash-lite",
  "judge_model": "gemini/gemini-2.5-flash-lite",
  "judge_family": "gemini",
  "trials": 1,
  "temperature": 0.0,
  "ranking_condition": "baseline",
  "timestamp_utc": "2026-05-06T09:42:07+00:00",
  "runner_git_commit": "854553c217be2dc712ea860fe695977144318e4e",
  "random_seed": null,
  "task_set": "main",
  "camera_injection": true,
  "judge_prompt_sha256": "b2289bcc4da706408132b89a4dbe1cdcd83423921779f69adb6eab8e2a03bec5",
  "judge_family_resolution": "explicit",
  "ranking_judge_model": null,
  "ranking_judge_family": null,
  "enable_repair": false,
  "manifest_warnings": []
}
```
