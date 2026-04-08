"""
ClinicalEnvironment — implements OpenEnv step() / reset() / state() interface.
"""
from __future__ import annotations
import uuid
from env.models import (
    PatientAction, TriageObservation, ClinicalState, TaskName,
    Task1Observation, Task2Observation, Task3Observation
)
from env.scenarios import get_task1_scenario, get_task2_scenario, get_task3_scenario
from env.graders import grade_task1, grade_task2, grade_task3

TASK_SEQUENCE = [
    TaskName.SINGLE_SYMPTOM_TRIAGE,
    TaskName.DIFFERENTIAL_DIAGNOSIS,
    TaskName.ICU_RESOURCE_ALLOCATION,
]

class OpenEnv:
    def __init__(self):
        self._state: ClinicalState | None = None
        self._current_patient = None
        self._gold_label: str | None = None
        self._task_index: int = 0

    def list_tasks(self) -> list[str]:
        """Return all task IDs — required by validator."""
        return [t.value for t in TASK_SEQUENCE]

    def reset(self) -> TriageObservation:
        obs = self._start_task()
        # Increment so the next /reset call yields the next task in the sequence
        self._task_index = (self._task_index + 1) % len(TASK_SEQUENCE)
        return obs

    def _start_task(self) -> TriageObservation:
        task = TASK_SEQUENCE[self._task_index]

        if task == TaskName.SINGLE_SYMPTOM_TRIAGE:
            patient, gold = get_task1_scenario()
            self._current_patient = patient
            self._gold_label = gold
            obs = TriageObservation(
                task_name=task, step_number=0,
                task1=Task1Observation(patient=patient)
            )

        elif task == TaskName.DIFFERENTIAL_DIAGNOSIS:
            patient, gold = get_task2_scenario()
            self._current_patient = patient
            self._gold_label = gold
            obs = TriageObservation(
                task_name=task, step_number=0,
                task2=Task2Observation(patient=patient)
            )

        else:  # ICU_RESOURCE_ALLOCATION
            patients, gold = get_task3_scenario()
            self._current_patient = patients
            self._gold_label = gold
            obs = TriageObservation(
                task_name=task, step_number=0,
                task3=Task3Observation(patients=patients)
            )

        self._state = ClinicalState(
            episode_id=str(uuid.uuid4()),
            current_task=task,
            step_count=0,
            max_steps=1,
            cumulative_reward=0.0,
            is_done=False,
        )
        return obs

    def step(self, action: PatientAction) -> TriageObservation:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        task = self._state.current_task

        if task == TaskName.SINGLE_SYMPTOM_TRIAGE:
            reward = grade_task1(action, self._gold_label)
            feedback = (f"Gold: {self._gold_label}. "
                        f"You chose: {action.task1.triage_level.value if action.task1 else 'N/A'}. "
                        f"Score: {reward.total:.2f}")
            obs = TriageObservation(
                task_name=task, step_number=1,
                task1=Task1Observation(
                    patient=self._current_patient,
                    feedback=feedback, reward=reward, done=True
                )
            )

        elif task == TaskName.DIFFERENTIAL_DIAGNOSIS:
            reward = grade_task2(action, self._gold_label)
            feedback = f"Score: {reward.total:.2f}"
            obs = TriageObservation(
                task_name=task, step_number=1,
                task2=Task2Observation(
                    patient=self._current_patient,
                    feedback=feedback, reward=reward, done=True
                )
            )

        else:
            reward = grade_task3(action, self._gold_label)
            feedback = f"Gold ICU: {self._gold_label}. Score: {reward.total:.2f}"
            obs = TriageObservation(
                task_name=task, step_number=1,
                task3=Task3Observation(
                    patients=self._current_patient,
                    feedback=feedback, reward=reward, done=True
                )
            )

        self._state.step_count += 1
        self._state.cumulative_reward += reward.total
        self._state.last_reward_breakdown = reward
        self._state.task_scores[task.value] = reward.total
        self._state.is_done = True

        return obs

    def state(self) -> ClinicalState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def grade(self, task_id: str, observations: any, action: PatientAction) -> float:
        """Standalone grade method — return clipped score."""
        if task_id == "single_symptom_triage":
            reward = grade_task1(action, self._gold_label)
        elif task_id == "differential_diagnosis":
            reward = grade_task2(action, self._gold_label)
        else:
            reward = grade_task3(action, self._gold_label)
        
        return max(0.001, min(0.999, float(reward.total)))
