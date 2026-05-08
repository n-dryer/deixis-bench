# Wearable Assistant Context Bench: Findings

**Benchmark:** situated context-tracking benchmark for AI wearable assistants used actively for advice or coaching

## Benchmark summary

- **Benchmark**: situated context-tracking benchmark for AI wearable assistants used actively for advice or coaching
- **Default comparison condition**: `baseline`
- **Primary score** - `mean(current_recall, prior_recall)` (class recall, not overall accuracy): **69.9% (95% CI 60.6%-79.2%)**
- **Bootstrap 95% CI on primary score**: 69.9% (95% CI 60.2%-79.4%) (5000 percentile bootstrap iterations)
- **How to read this run**: compare candidate models on the `baseline` score below; treat the other conditions as diagnostic sensitivity checks. CIs are 95% Wilson per class and 95% normal-approximation on the mean recall (with a bootstrap second opinion).
- **Per-class recall under `baseline`** (TP / (TP + FN)):
    - `current_recall`: 84.8% (95% CI 71.8%-92.4%)
    - `prior_recall`: 55.0% (95% CI 39.8%-69.3%)

Condition sensitivity (mean per-class recall):

| Condition | Mean recall (95% CI) |
| --- | --- |
| baseline (default) | 69.9% (95% CI 60.6%-79.2%) |
| context_selection_instruction | 80.5% (95% CI 72.4%-88.6%) |
| pre_answer_context_scaffold | 93.9% (95% CI 88.8%-99.0%) |

## Per task-set recall

| Task set | Mean recall (95% CI) |
| --- | --- |
| `main` | 69.9% (95% CI 60.6%-79.2%) |

## Per shift-type recall

| Shift type | Pass rate (95% CI) |
| --- | --- |
| `absent_referent` | 47.6% (95% CI 28.3%-67.6%) |
| `cross_session_reference` | 85.0% (95% CI 64.0%-94.8%) |
| `location` | 40.9% (95% CI 23.3%-61.3%) |
| `object_in_hand` | 71.4% (95% CI 50.0%-86.2%) |
| `object_in_view` | 57.1% (95% CI 36.5%-75.5%) |
| `object_state` | 54.5% (95% CI 34.7%-73.1%) |
| `screen_content` | 61.9% (95% CI 40.9%-79.2%) |
| `sequential_task` | 72.2% (95% CI 49.1%-87.5%) |

## Per-class pass rate by condition

| Class | baseline | context_selection_instruction | pre_answer_context_scaffold |
| --- | --- | --- | --- |
| `current` | 84.8% [95% CI 71.8-92.4] (39/46) | 93.5% [95% CI 82.5-97.8] (43/46) | 97.8% [95% CI 88.7-99.6] (45/46) |
| `prior` | 55.0% [95% CI 39.8-69.3] (22/40) | 67.5% [95% CI 52.0-79.9] (27/40) | 90.0% [95% CI 76.9-96.0] (36/40) |
| `clarify` (auxiliary) | 32.5% [95% CI 20.1-48.0] (13/40) | 27.5% [95% CI 16.1-42.8] (11/40) | 5.0% [95% CI 1.4-16.5] (2/40) |
| `abstain` (auxiliary) | 67.5% [95% CI 52.0-79.9] (27/40) | 67.5% [95% CI 52.0-79.9] (27/40) | 25.0% [95% CI 14.2-40.2] (10/40) |

## Hedging behavior

- **Clarification rate**: 14.5% (95% CI 9.9%-20.6%)
- **Abstention rate**: 18.7% (95% CI 13.5%-25.3%)
- **Coverage** (1 - clarify - abstain): 66.9% (95% CI 59.4%-73.6%)

## Code-judge disagreement by task

- task-001: 0 trial(s) with code/judge disagreement
- task-002: 0 trial(s) with code/judge disagreement
- task-003: 0 trial(s) with code/judge disagreement
- task-004: 0 trial(s) with code/judge disagreement
- task-005: 0 trial(s) with code/judge disagreement
- task-006: 0 trial(s) with code/judge disagreement
- task-007: 0 trial(s) with code/judge disagreement
- task-008: 1 trial(s) with code/judge disagreement
- task-009: 2 trial(s) with code/judge disagreement
- task-010: 0 trial(s) with code/judge disagreement
- task-011: 0 trial(s) with code/judge disagreement
- task-012: 1 trial(s) with code/judge disagreement
- task-013: 0 trial(s) with code/judge disagreement
- task-014: 0 trial(s) with code/judge disagreement
- task-015: 0 trial(s) with code/judge disagreement
- task-016: 0 trial(s) with code/judge disagreement
- task-017: 1 trial(s) with code/judge disagreement
- task-018: 0 trial(s) with code/judge disagreement
- task-019: 0 trial(s) with code/judge disagreement
- task-020: 0 trial(s) with code/judge disagreement
- task-021: 0 trial(s) with code/judge disagreement
- task-022: 0 trial(s) with code/judge disagreement
- task-023: 0 trial(s) with code/judge disagreement
- task-024: 2 trial(s) with code/judge disagreement
- task-025: 0 trial(s) with code/judge disagreement
- task-026: 1 trial(s) with code/judge disagreement
- task-027: 0 trial(s) with code/judge disagreement
- task-028: 0 trial(s) with code/judge disagreement
- task-029: 0 trial(s) with code/judge disagreement
- task-030: 0 trial(s) with code/judge disagreement
- task-031: 1 trial(s) with code/judge disagreement
- task-032: 2 trial(s) with code/judge disagreement
- task-033: 0 trial(s) with code/judge disagreement
- task-034: 0 trial(s) with code/judge disagreement
- task-035: 0 trial(s) with code/judge disagreement
- task-036: 2 trial(s) with code/judge disagreement
- task-037: 0 trial(s) with code/judge disagreement
- task-038: 1 trial(s) with code/judge disagreement
- task-039: 0 trial(s) with code/judge disagreement
- task-040: 0 trial(s) with code/judge disagreement
- task-041: 3 trial(s) with code/judge disagreement
- task-042: 1 trial(s) with code/judge disagreement
- task-043: 0 trial(s) with code/judge disagreement
- task-044: 0 trial(s) with code/judge disagreement
- task-045: 0 trial(s) with code/judge disagreement
- task-046: 1 trial(s) with code/judge disagreement
- task-047: 1 trial(s) with code/judge disagreement
- task-048: 0 trial(s) with code/judge disagreement
- task-049: 0 trial(s) with code/judge disagreement
- task-050: 0 trial(s) with code/judge disagreement
- task-051: 1 trial(s) with code/judge disagreement
- task-052: 1 trial(s) with code/judge disagreement
- task-053: 0 trial(s) with code/judge disagreement
- task-054: 0 trial(s) with code/judge disagreement
- task-055: 0 trial(s) with code/judge disagreement
- task-056: 0 trial(s) with code/judge disagreement
- task-057: 0 trial(s) with code/judge disagreement
- task-058: 0 trial(s) with code/judge disagreement
- task-059: 0 trial(s) with code/judge disagreement
- task-060: 0 trial(s) with code/judge disagreement
- task-061: 2 trial(s) with code/judge disagreement
- task-062: 2 trial(s) with code/judge disagreement
- task-063: 2 trial(s) with code/judge disagreement
- task-064: 0 trial(s) with code/judge disagreement
- task-065: 0 trial(s) with code/judge disagreement
- task-066: 0 trial(s) with code/judge disagreement
- task-067: 1 trial(s) with code/judge disagreement
- task-068: 0 trial(s) with code/judge disagreement
- task-069: 1 trial(s) with code/judge disagreement
- task-070: 0 trial(s) with code/judge disagreement
- task-071: 0 trial(s) with code/judge disagreement
- task-072: 0 trial(s) with code/judge disagreement
- task-073: 0 trial(s) with code/judge disagreement
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
- task-086: 1 trial(s) with code/judge disagreement
- task-087: 0 trial(s) with code/judge disagreement
- task-088: 0 trial(s) with code/judge disagreement
- task-089: 1 trial(s) with code/judge disagreement
- task-090: 0 trial(s) with code/judge disagreement
- task-091: 0 trial(s) with code/judge disagreement
- task-092: 1 trial(s) with code/judge disagreement
- task-093: 0 trial(s) with code/judge disagreement
- task-094: 1 trial(s) with code/judge disagreement
- task-095: 0 trial(s) with code/judge disagreement
- task-096: 2 trial(s) with code/judge disagreement
- task-097: 0 trial(s) with code/judge disagreement
- task-098: 2 trial(s) with code/judge disagreement
- task-099: 0 trial(s) with code/judge disagreement
- task-100: 1 trial(s) with code/judge disagreement
- task-101: 0 trial(s) with code/judge disagreement
- task-102: 0 trial(s) with code/judge disagreement
- task-103: 0 trial(s) with code/judge disagreement
- task-104: 0 trial(s) with code/judge disagreement
- task-105: 1 trial(s) with code/judge disagreement
- task-106: 1 trial(s) with code/judge disagreement
- task-107: 0 trial(s) with code/judge disagreement
- task-108: 1 trial(s) with code/judge disagreement
- task-109: 1 trial(s) with code/judge disagreement
- task-110: 3 trial(s) with code/judge disagreement
- task-111: 0 trial(s) with code/judge disagreement
- task-112: 0 trial(s) with code/judge disagreement
- task-113: 0 trial(s) with code/judge disagreement
- task-114: 0 trial(s) with code/judge disagreement
- task-115: 0 trial(s) with code/judge disagreement
- task-116: 0 trial(s) with code/judge disagreement
- task-117: 0 trial(s) with code/judge disagreement
- task-118: 0 trial(s) with code/judge disagreement
- task-119: 2 trial(s) with code/judge disagreement
- task-120: 2 trial(s) with code/judge disagreement
- task-121: 3 trial(s) with code/judge disagreement
- task-122: 1 trial(s) with code/judge disagreement
- task-123: 1 trial(s) with code/judge disagreement
- task-124: 0 trial(s) with code/judge disagreement
- task-125: 0 trial(s) with code/judge disagreement
- task-126: 0 trial(s) with code/judge disagreement
- task-127: 1 trial(s) with code/judge disagreement
- task-128: 2 trial(s) with code/judge disagreement
- task-129: 0 trial(s) with code/judge disagreement
- task-130: 0 trial(s) with code/judge disagreement
- task-131: 2 trial(s) with code/judge disagreement
- task-132: 0 trial(s) with code/judge disagreement
- task-133: 0 trial(s) with code/judge disagreement
- task-134: 1 trial(s) with code/judge disagreement
- task-135: 1 trial(s) with code/judge disagreement
- task-136: 0 trial(s) with code/judge disagreement
- task-137: 1 trial(s) with code/judge disagreement
- task-138: 3 trial(s) with code/judge disagreement
- task-139: 1 trial(s) with code/judge disagreement
- task-140: 0 trial(s) with code/judge disagreement
- task-141: 0 trial(s) with code/judge disagreement
- task-142: 0 trial(s) with code/judge disagreement
- task-143: 0 trial(s) with code/judge disagreement
- task-144: 0 trial(s) with code/judge disagreement
- task-145: 0 trial(s) with code/judge disagreement
- task-146: 1 trial(s) with code/judge disagreement
- task-147: 0 trial(s) with code/judge disagreement
- task-148: 0 trial(s) with code/judge disagreement
- task-149: 0 trial(s) with code/judge disagreement
- task-150: 0 trial(s) with code/judge disagreement
- task-151: 0 trial(s) with code/judge disagreement
- task-152: 0 trial(s) with code/judge disagreement
- task-153: 0 trial(s) with code/judge disagreement
- task-154: 0 trial(s) with code/judge disagreement
- task-155: 0 trial(s) with code/judge disagreement
- task-156: 0 trial(s) with code/judge disagreement
- task-157: 1 trial(s) with code/judge disagreement
- task-158: 0 trial(s) with code/judge disagreement
- task-159: 0 trial(s) with code/judge disagreement
- task-160: 1 trial(s) with code/judge disagreement
- task-161: 0 trial(s) with code/judge disagreement
- task-162: 0 trial(s) with code/judge disagreement
- task-163: 0 trial(s) with code/judge disagreement
- task-164: 0 trial(s) with code/judge disagreement
- task-165: 0 trial(s) with code/judge disagreement
- task-166: 0 trial(s) with code/judge disagreement

## Inter-judge agreement (cross-LLM)

_No ranking-judge labels in this run. To enable cross-LLM inter-judge agreement, pass `--ranking-judge-family` to the runner so every trial is also labeled by a fixed second judge._

## Task-by-condition matrix

| Task | Target context | baseline | context_selection_instruction | pre_answer_context_scaffold |
| --- | --- | --- | --- | --- |
| task-001 | `current` | pass | pass | pass |
| task-002 | `current` | pass | pass | pass |
| task-003 | `current` | pass | pass | pass |
| task-004 | `prior` | pass | pass | pass |
| task-005 | `current` | pass | pass | pass |
| task-006 | `current` | pass | pass | pass |
| task-007 | `current` | pass | pass | pass |
| task-008 | `current` | pass | pass | pass |
| task-009 | `clarify` | pass | fail | pass |
| task-010 | `current` | pass | pass | pass |
| task-011 | `current` | pass | pass | pass |
| task-012 | `prior` | fail | fail | pass |
| task-013 | `current` | pass | pass | pass |
| task-014 | `current` | pass | pass | pass |
| task-015 | `current` | pass | pass | pass |
| task-016 | `prior` | pass | pass | pass |
| task-017 | `current` | fail | pass | pass |
| task-018 | `current` | pass | pass | pass |
| task-019 | `prior` | fail | pass | pass |
| task-020 | `current` | pass | pass | pass |
| task-021 | `current` | pass | pass | pass |
| task-022 | `current` | pass | pass | pass |
| task-023 | `current` | pass | pass | pass |
| task-024 | `prior` | pass | pass | pass |
| task-025 | `current` | pass | pass | pass |
| task-026 | `current` | pass | pass | pass |
| task-027 | `current` | pass | pass | pass |
| task-028 | `current` | pass | pass | pass |
| task-029 | `current` | pass | pass | pass |
| task-030 | `current` | pass | pass | pass |
| task-031 | `prior` | fail | fail | pass |
| task-032 | `clarify` | fail | fail | fail |
| task-033 | `current` | pass | pass | pass |
| task-034 | `current` | pass | pass | pass |
| task-035 | `current` | pass | pass | pass |
| task-036 | `current` | pass | pass | pass |
| task-037 | `prior` | fail | pass | pass |
| task-038 | `prior` | fail | pass | pass |
| task-039 | `abstain` | pass | pass | fail |
| task-040 | `current` | pass | pass | pass |
| task-041 | `clarify` | fail | fail | fail |
| task-042 | `abstain` | fail | pass | fail |
| task-043 | `current` | fail | pass | pass |
| task-044 | `current` | pass | pass | pass |
| task-045 | `current` | fail | pass | pass |
| task-046 | `prior` | fail | pass | pass |
| task-047 | `prior` | pass | fail | pass |
| task-048 | `prior` | pass | pass | pass |
| task-049 | `prior` | pass | fail | pass |
| task-050 | `current` | pass | pass | pass |
| task-051 | `current` | fail | fail | pass |
| task-052 | `current` | pass | pass | pass |
| task-053 | `current` | pass | pass | pass |
| task-054 | `current` | pass | pass | pass |
| task-055 | `current` | pass | pass | pass |
| task-056 | `current` | pass | pass | pass |
| task-057 | `current` | pass | pass | pass |
| task-058 | `current` | pass | pass | pass |
| task-059 | `current` | pass | pass | pass |
| task-060 | `current` | pass | pass | pass |
| task-061 | `current` | fail | fail | fail |
| task-062 | `current` | fail | pass | pass |
| task-063 | `current` | fail | fail | pass |
| task-064 | `prior` | fail | fail | fail |
| task-065 | `prior` | pass | pass | pass |
| task-066 | `prior` | fail | pass | pass |
| task-067 | `prior` | pass | pass | pass |
| task-068 | `clarify` | fail | pass | fail |
| task-069 | `clarify` | fail | fail | fail |
| task-070 | `abstain` | fail | fail | fail |
| task-071 | `prior` | pass | pass | pass |
| task-072 | `clarify` | fail | fail | fail |
| task-073 | `abstain` | fail | fail | fail |
| task-074 | `prior` | pass | pass | fail |
| task-075 | `prior` | pass | pass | pass |
| task-076 | `prior` | pass | pass | pass |
| task-077 | `clarify` | fail | fail | fail |
| task-078 | `clarify` | fail | fail | fail |
| task-079 | `clarify` | fail | fail | fail |
| task-080 | `clarify` | fail | fail | fail |
| task-081 | `clarify` | fail | fail | fail |
| task-082 | `abstain` | fail | fail | fail |
| task-083 | `abstain` | fail | fail | fail |
| task-084 | `prior` | fail | fail | pass |
| task-085 | `prior` | fail | pass | pass |
| task-086 | `clarify` | pass | pass | fail |
| task-087 | `clarify` | fail | pass | fail |
| task-088 | `clarify` | pass | pass | fail |
| task-089 | `clarify` | pass | pass | fail |
| task-090 | `clarify` | pass | pass | fail |
| task-091 | `clarify` | fail | fail | fail |
| task-092 | `clarify` | pass | fail | fail |
| task-093 | `clarify` | fail | fail | fail |
| task-094 | `clarify` | fail | fail | fail |
| task-095 | `abstain` | pass | pass | pass |
| task-096 | `prior` | pass | fail | fail |
| task-097 | `prior` | pass | fail | pass |
| task-098 | `prior` | fail | fail | fail |
| task-099 | `clarify` | pass | fail | fail |
| task-100 | `clarify` | fail | pass | fail |
| task-101 | `clarify` | pass | fail | fail |
| task-102 | `clarify` | fail | fail | fail |
| task-103 | `clarify` | fail | fail | fail |
| task-104 | `abstain` | pass | pass | pass |
| task-105 | `abstain` | fail | pass | pass |
| task-106 | `abstain` | fail | pass | fail |
| task-107 | `abstain` | pass | pass | pass |
| task-108 | `prior` | pass | pass | pass |
| task-109 | `prior` | fail | pass | pass |
| task-110 | `prior` | fail | fail | pass |
| task-111 | `clarify` | fail | fail | fail |
| task-112 | `clarify` | fail | fail | fail |
| task-113 | `clarify` | fail | fail | fail |
| task-114 | `clarify` | fail | fail | fail |
| task-115 | `clarify` | fail | fail | fail |
| task-116 | `clarify` | fail | fail | fail |
| task-117 | `clarify` | fail | fail | fail |
| task-118 | `abstain` | pass | pass | fail |
| task-119 | `abstain` | fail | fail | fail |
| task-120 | `abstain` | pass | pass | fail |
| task-121 | `prior` | fail | fail | pass |
| task-122 | `prior` | fail | fail | pass |
| task-123 | `prior` | fail | fail | pass |
| task-124 | `prior` | fail | pass | pass |
| task-125 | `clarify` | fail | fail | fail |
| task-126 | `clarify` | pass | pass | pass |
| task-127 | `clarify` | pass | fail | fail |
| task-128 | `clarify` | fail | pass | fail |
| task-129 | `abstain` | pass | fail | fail |
| task-130 | `abstain` | pass | pass | fail |
| task-131 | `abstain` | pass | fail | fail |
| task-132 | `abstain` | pass | pass | pass |
| task-133 | `abstain` | fail | fail | fail |
| task-134 | `abstain` | pass | fail | fail |
| task-135 | `abstain` | pass | pass | fail |
| task-136 | `prior` | pass | pass | pass |
| task-137 | `prior` | fail | pass | pass |
| task-138 | `prior` | pass | pass | pass |
| task-139 | `clarify` | pass | fail | fail |
| task-140 | `clarify` | pass | pass | fail |
| task-141 | `clarify` | fail | pass | fail |
| task-142 | `clarify` | pass | fail | fail |
| task-143 | `abstain` | pass | pass | fail |
| task-144 | `abstain` | fail | fail | fail |
| task-145 | `abstain` | pass | pass | fail |
| task-146 | `abstain` | pass | pass | fail |
| task-147 | `abstain` | pass | pass | pass |
| task-148 | `abstain` | pass | pass | pass |
| task-149 | `abstain` | pass | fail | fail |
| task-150 | `abstain` | pass | pass | fail |
| task-151 | `prior` | pass | pass | pass |
| task-152 | `prior` | pass | pass | pass |
| task-153 | `prior` | pass | pass | pass |
| task-154 | `prior` | pass | pass | pass |
| task-155 | `prior` | pass | pass | pass |
| task-156 | `abstain` | fail | pass | fail |
| task-157 | `abstain` | pass | pass | fail |
| task-158 | `abstain` | pass | pass | fail |
| task-159 | `abstain` | pass | pass | fail |
| task-160 | `abstain` | pass | fail | pass |
| task-161 | `abstain` | pass | pass | pass |
| task-162 | `abstain` | fail | fail | fail |
| task-163 | `abstain` | pass | pass | pass |
| task-164 | `abstain` | pass | pass | fail |
| task-165 | `abstain` | pass | pass | fail |
| task-166 | `abstain` | fail | pass | fail |

## Reproducibility manifest

```json
{
  "benchmark_version": "0.1.0a0",
  "tasks_sha256": "bd3611d20996a3b3d6a6a1d9c9833f36b5a287bb4f746d26cec2537777b157d1",
  "prompt_conditions_sha256": "292f7dc0631c956c850ba868106c395f1d0865589ebd123df1a5471095b77bce",
  "candidate_model": "gemini/gemini-2.5-flash",
  "judge_model": "gemini/gemini-2.5-flash-lite",
  "judge_family": "gemini",
  "trials": 1,
  "temperature": 0.0,
  "ranking_condition": "baseline",
  "timestamp_utc": "2026-05-06T20:01:04+00:00",
  "runner_git_commit": "b31185f820fbc6bbb36dc11231ad87af9514a062",
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
