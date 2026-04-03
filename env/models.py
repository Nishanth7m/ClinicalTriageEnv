from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class TriageLevel(str, Enum):
    EMERGENT   = "emergent"
    URGENT     = "urgent"
    NON_URGENT = "non_urgent"

class TaskName(str, Enum):
    SINGLE_SYMPTOM_TRIAGE   = "single_symptom_triage"
    DIFFERENTIAL_DIAGNOSIS  = "differential_diagnosis"
    ICU_RESOURCE_ALLOCATION = "icu_resource_allocation"

class VitalSigns(BaseModel):
    heart_rate_bpm:       int   = Field(..., ge=0, le=300)
    systolic_bp_mmhg:     int   = Field(..., ge=0, le=300)
    diastolic_bp_mmhg:    int   = Field(..., ge=0, le=200)
    spo2_percent:         float = Field(..., ge=0.0, le=100.0)
    respiratory_rate_rpm: int   = Field(..., ge=0, le=60)
    temperature_celsius:  float = Field(..., ge=30.0, le=45.0)
    glasgow_coma_scale:   int   = Field(..., ge=3, le=15)

class LabResult(BaseModel):
    test_name:    str
    value:        float
    unit:         str
    reference_lo: Optional[float] = None
    reference_hi: Optional[float] = None

class PatientRecord(BaseModel):
    patient_id:      str
    age_years:       int
    sex:             str
    chief_complaint: str
    symptoms:        List[str]
    vitals:          VitalSigns
    labs:            List[LabResult] = []
    history:         List[str] = []
    medications:     List[str] = []
    allergies:       List[str] = []
    arrival_mode:    str = "walk_in"

class RewardBreakdown(BaseModel):
    triage_accuracy:   float = 0.0
    emergent_penalty:  float = 0.0
    guideline_bonus:   float = 0.0
    reasoning_quality: float = 0.0
    total:             float = 0.0

    def compute_total(self) -> float:
        raw = (self.triage_accuracy + self.emergent_penalty
               + self.guideline_bonus + self.reasoning_quality)
        self.total = max(-1.0, min(1.0, raw))
        return self.total

class Task1Action(BaseModel):
    triage_level:     TriageLevel
    rationale:        str = Field(..., min_length=10, max_length=500)
    cited_guideline:  Optional[str] = None

class Task2Action(BaseModel):
    primary_diagnosis:       str
    differential_diagnoses:  List[str] = Field(..., min_length=2)
    recommended_actions:     List[str] = Field(..., min_length=1)
    triage_level:            TriageLevel
    reasoning:               str = Field(..., min_length=20)
    cited_guidelines:        List[str] = []

class ICUAllocationDecision(BaseModel):
    patient_id:    str
    allocated_icu: bool
    priority_rank: int
    justification: str = Field(..., min_length=10)

class Task3Action(BaseModel):
    allocation_decisions: List[ICUAllocationDecision]
    overall_reasoning:    str = Field(..., min_length=30)
    cited_guidelines:     List[str] = []

class PatientAction(BaseModel):
    task1: Optional[Task1Action] = None
    task2: Optional[Task2Action] = None
    task3: Optional[Task3Action] = None

class Task1Observation(BaseModel):
    patient:  PatientRecord
    feedback: Optional[str] = None
    reward:   Optional[RewardBreakdown] = None
    done:     bool = False

class Task2Observation(BaseModel):
    patient:  PatientRecord
    feedback: Optional[str] = None
    reward:   Optional[RewardBreakdown] = None
    done:     bool = False

class Task3Observation(BaseModel):
    patients:           List[PatientRecord]
    available_icu_beds: int = 1
    feedback:           Optional[str] = None
    reward:             Optional[RewardBreakdown] = None
    done:               bool = False

class TriageObservation(BaseModel):
    task_name:   TaskName
    step_number: int = 0
    task1:       Optional[Task1Observation] = None
    task2:       Optional[Task2Observation] = None
    task3:       Optional[Task3Observation] = None
    info:        Dict[str, str] = {}

class ClinicalState(BaseModel):
    episode_id:            str
    current_task:          TaskName
    step_count:            int   = 0
    max_steps:             int   = 1
    cumulative_reward:     float = 0.0
    is_done:               bool  = False
    last_reward_breakdown: Optional[RewardBreakdown] = None
    task_scores:           Dict[str, float] = {}
