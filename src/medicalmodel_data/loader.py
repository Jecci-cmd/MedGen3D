from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TASK_IDS = {"segmentation": 0, "restoration": 1, "reconstruction": 2}


@dataclass(frozen=True)
class Sample:
    source: np.ndarray
    target: np.ndarray
    instruction: str
    task: str
    task_id: int
    case_id: str
    sample_id: str


def load_sample(record: dict[str, object]) -> Sample:
    source_spec = record["source"]
    target_spec = record["target"]
    arrays = np.load(source_spec["path"])
    source = arrays[source_spec["key"]].astype(np.float32, copy=False)
    target = arrays[target_spec["key"]].astype(np.float32, copy=False)
    if source.ndim == 3:
        source = source[None, ...]
    if target.ndim == 3:
        target = target[None, ...]
    task = str(record["task"])
    return Sample(
        source=source,
        target=target,
        instruction=str(record["instruction"]),
        task=task,
        task_id=TASK_IDS[task],
        case_id=str(record["case_id"]),
        sample_id=str(record["sample_id"]),
    )


def collate_same_task(samples: list[Sample]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    tasks = {sample.task for sample in samples}
    if len(tasks) != 1:
        raise ValueError("Same-task collation received mixed tasks")
    return {
        "source": np.stack([sample.source for sample in samples]),
        "target": np.stack([sample.target for sample in samples]),
        "instruction": [sample.instruction for sample in samples],
        "task_id": np.asarray([sample.task_id for sample in samples], dtype=np.int64),
        "case_id": [sample.case_id for sample in samples],
        "sample_id": [sample.sample_id for sample in samples],
    }


def make_task_balanced_batch(samples: list[Sample]) -> dict[str, object]:
    by_task: dict[str, list[Sample]] = {task: [] for task in TASK_IDS}
    for sample in samples:
        by_task[sample.task].append(sample)
    if any(not task_samples for task_samples in by_task.values()):
        missing = [task for task, values in by_task.items() if not values]
        raise ValueError(f"Task-balanced batch missing tasks: {missing}")
    selected = [by_task[task][0] for task in TASK_IDS]
    return {
        "samples": selected,
        "instruction": [sample.instruction for sample in selected],
        "task_id": np.asarray([sample.task_id for sample in selected], dtype=np.int64),
        "case_id": [sample.case_id for sample in selected],
        "sample_id": [sample.sample_id for sample in selected],
        "per_task": {
            task: collate_same_task([sample]) for task, sample in zip(TASK_IDS, selected)
        },
    }
