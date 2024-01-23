from datetime import datetime

from psycop.common.data_structures import TemporalEvent
from psycop.common.data_structures.patient import Patient


def create_patients() -> list[Patient]:
    e1 = TemporalEvent(
        timestamp=datetime(2021, 1, 1), value="I65", source_type="diagnosis", source_subtype="A"
    )
    e2 = TemporalEvent(
        timestamp=datetime(2021, 1, 3), value="A30", source_type="diagnosis", source_subtype="A"
    )

    patient1 = Patient(patient_id=1, date_of_birth=datetime(1990, 1, 1))
    patient1.add_events([e1, e2])

    patient2 = Patient(patient_id=2, date_of_birth=datetime(1993, 3, 1))
    patient2.add_events([e1, e2, e2, e1])
    return [patient1, patient2]
