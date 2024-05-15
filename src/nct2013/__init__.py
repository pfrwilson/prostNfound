CENTERS = ["UVA", "PCC", "PMCC", "CRCEO", "JH"]
CORE_LOCATIONS = [
    "LML",
    "RBL",
    "LMM",
    "RMM",
    "LBL",
    "LAM",
    "RAM",
    "RML",
    "LBM",
    "RAL",
    "RBM",
    "LAL",
]

GRADE_GROUP_VALUE_COUNTS = {0: 5727, 2: 480, 3: 195, 4: 134, 5: 71}


from enum import StrEnum

class MetadataKeys(StrEnum): 
    CENTER = 'center'
    LOC = 'loc'
    GRADE = 'grade'
    AGE = 'age'
    FAMILY_HISTORY = 'family_history'
    PSA = 'psa'
    PCT_CANCER = 'pct_cancer'
    PRIMARY_GRADE = 'primary_grade'
    SECONDARY_GRADE = 'secondary_grade'
    PATIENT_ID = 'patient_id'
    CORE_ID = 'core_id'
    ALL_CORES_BENIGN = 'all_cores_benign'
    GRADE_GROUP = 'grade_group'
    CLINICALLY_SIGNIFICANT = 'clinically_significant'
    APPROX_PSA_DENSITY = 'approx_psa_density'


from .cohort_selection import select_cohort
from .data_access import data_accessor
from .dataset import DataKeys, NCT2013Dataset
from .helpers import *