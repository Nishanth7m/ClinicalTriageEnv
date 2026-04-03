"""Static scenario bank — returns a random PatientRecord for each task."""
import random
from env.models import PatientRecord, VitalSigns, LabResult

TASK1_SCENARIOS = [
    # (patient_dict, gold_label)
    (dict(
        patient_id="P001", age_years=58, sex="M",
        chief_complaint="crushing chest pain radiating to left arm",
        symptoms=["chest pain", "diaphoresis", "nausea", "shortness of breath"],
        vitals=VitalSigns(heart_rate_bpm=110, systolic_bp_mmhg=88,
                          diastolic_bp_mmhg=60, spo2_percent=94.0,
                          respiratory_rate_rpm=22, temperature_celsius=37.1,
                          glasgow_coma_scale=15),
        labs=[LabResult(test_name="troponin_I", value=2.4, unit="ng/mL",
                        reference_lo=0.0, reference_hi=0.04)],
        history=["hypertension", "type 2 diabetes"],
        medications=["metformin", "lisinopril"], allergies=[],
        arrival_mode="ambulance"
    ), "emergent"),
    (dict(
        patient_id="P002", age_years=34, sex="F",
        chief_complaint="fever and productive cough for 3 days",
        symptoms=["fever", "cough", "fatigue", "mild dyspnoea"],
        vitals=VitalSigns(heart_rate_bpm=95, systolic_bp_mmhg=118,
                          diastolic_bp_mmhg=76, spo2_percent=96.5,
                          respiratory_rate_rpm=18, temperature_celsius=38.8,
                          glasgow_coma_scale=15),
        labs=[LabResult(test_name="wbc", value=14.2, unit="10^9/L",
                        reference_lo=4.5, reference_hi=11.0)],
        history=[], medications=[], allergies=["penicillin"],
        arrival_mode="walk_in"
    ), "urgent"),
    (dict(
        patient_id="P003", age_years=22, sex="F",
        chief_complaint="sprained ankle after sport",
        symptoms=["ankle pain", "mild swelling", "difficulty weight-bearing"],
        vitals=VitalSigns(heart_rate_bpm=72, systolic_bp_mmhg=120,
                          diastolic_bp_mmhg=78, spo2_percent=99.0,
                          respiratory_rate_rpm=14, temperature_celsius=36.6,
                          glasgow_coma_scale=15),
        labs=[], history=[], medications=[], allergies=[],
        arrival_mode="walk_in"
    ), "non_urgent"),
    (dict(
        patient_id="P004", age_years=67, sex="M",
        chief_complaint="sudden severe headache, worst of life",
        symptoms=["thunderclap headache", "neck stiffness", "photophobia", "vomiting"],
        vitals=VitalSigns(heart_rate_bpm=58, systolic_bp_mmhg=180,
                          diastolic_bp_mmhg=110, spo2_percent=97.0,
                          respiratory_rate_rpm=16, temperature_celsius=37.4,
                          glasgow_coma_scale=13),
        labs=[], history=["hypertension"], medications=["amlodipine"],
        allergies=[], arrival_mode="ambulance"
    ), "emergent"),
    (dict(
        patient_id="P005", age_years=45, sex="F",
        chief_complaint="abdominal pain and vomiting since morning",
        symptoms=["right lower quadrant pain", "nausea", "vomiting", "low-grade fever"],
        vitals=VitalSigns(heart_rate_bpm=88, systolic_bp_mmhg=126,
                          diastolic_bp_mmhg=80, spo2_percent=98.0,
                          respiratory_rate_rpm=16, temperature_celsius=37.9,
                          glasgow_coma_scale=15),
        labs=[LabResult(test_name="wbc", value=13.1, unit="10^9/L",
                        reference_lo=4.5, reference_hi=11.0)],
        history=[], medications=[], allergies=[], arrival_mode="walk_in"
    ), "urgent"),
]

TASK3_SCENARIOS = [
    [
        dict(patient_id="PA", age_years=72, sex="M",
             chief_complaint="septic shock post-op",
             symptoms=["hypotension", "fever", "altered consciousness", "oliguria"],
             vitals=VitalSigns(heart_rate_bpm=128, systolic_bp_mmhg=74,
                               diastolic_bp_mmhg=44, spo2_percent=91.0,
                               respiratory_rate_rpm=28, temperature_celsius=39.4,
                               glasgow_coma_scale=10),
             labs=[LabResult(test_name="lactate", value=6.2, unit="mmol/L", reference_lo=0.5, reference_hi=2.0),
                   LabResult(test_name="creatinine", value=3.8, unit="mg/dL", reference_lo=0.7, reference_hi=1.3)],
             history=["CKD", "hypertension"], medications=[], allergies=[],
             arrival_mode="ambulance"),
        dict(patient_id="PB", age_years=54, sex="F",
             chief_complaint="massive STEMI",
             symptoms=["chest pain", "diaphoresis", "pulmonary oedema"],
             vitals=VitalSigns(heart_rate_bpm=116, systolic_bp_mmhg=82,
                               diastolic_bp_mmhg=52, spo2_percent=88.0,
                               respiratory_rate_rpm=26, temperature_celsius=36.9,
                               glasgow_coma_scale=14),
             labs=[LabResult(test_name="troponin_I", value=18.5, unit="ng/mL", reference_lo=0.0, reference_hi=0.04)],
             history=["diabetes"], medications=["insulin"], allergies=[],
             arrival_mode="ambulance"),
        dict(patient_id="PC", age_years=38, sex="M",
             chief_complaint="severe asthma exacerbation",
             symptoms=["wheeze", "accessory muscle use", "cannot complete sentences"],
             vitals=VitalSigns(heart_rate_bpm=118, systolic_bp_mmhg=134,
                               diastolic_bp_mmhg=88, spo2_percent=89.0,
                               respiratory_rate_rpm=32, temperature_celsius=37.2,
                               glasgow_coma_scale=15),
             labs=[], history=["asthma"], medications=["salbutamol"], allergies=[],
             arrival_mode="ambulance"),
    ]
]
# Gold ICU allocation for Task3: PB (STEMI with cardiogenic shock) > PA > PC

def get_task1_scenario():
    item = random.choice(TASK1_SCENARIOS)
    return PatientRecord(**item[0]), item[1]

def get_task2_scenario():
    # Reuse task1 patients for task2 (same records, harder reasoning expected)
    item = random.choice(TASK1_SCENARIOS)
    return PatientRecord(**item[0]), item[1]

def get_task3_scenario():
    group = random.choice(TASK3_SCENARIOS)
    return [PatientRecord(**p) for p in group], "PB"  # gold = PB gets ICU
