# currently on step 2 of implement_opimize_plan.md (only implemented step 1 so far)

# The function will take in the student and teacher schedules and output the best time for the teacher to have office hours.
# The function will use the pymoo library to find the best time for the teacher to have office hours.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

# Step 2 constants for parsing schedule CSV files into slot arrays.
DAY_TO_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
DAYS_PER_WEEK = 5


def _global_slot(day_code: str, slot_in_day: int, slots_per_day: int) -> int:
    """Map (day, day-local slot index) to a global slot index."""
    day = day_code.strip().lower() #removes whitespace and converts to lowercase
    if day not in DAY_TO_INDEX: #checks if the day is valid
        raise ValueError(f"Unknown day code: {day_code}")
    if slot_in_day < 0: #checks if the slot is valid
        raise ValueError("slot_in_day must be >= 0.")
    return DAY_TO_INDEX[day] * slots_per_day + slot_in_day

# infers the number of slots per day from the csv files (LOOK INTO MORE)
def _infer_slots_per_day(
    teacher_csv_path: str | Path, student_csv_path: str | Path #path to the csv files
) -> int:
    """Infer slots_per_day from the max end_slot value in the CSV inputs."""
    max_end = 0

    with open(teacher_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            end_slot = int(row["end_slot"])
            max_end = max(max_end, end_slot)

    with open(student_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("end_slot"):
                continue
            end_slot = int(row["end_slot"])
            max_end = max(max_end, end_slot)

    if max_end <= 0:
        raise ValueError("Could not infer slots_per_day from CSV files.")
    return max_end

# loads the teacher's availability from the csv file
def load_teacher_availability_csv(
    teacher_csv_path: str | Path, slots_per_day: int
) -> np.ndarray:
    """
    Parse teacher CSV rows into a 1D availability vector.

    Expects columns: day,start_slot,end_slot
    Uses half-open intervals [start_slot, end_slot).
    """
    total_slots = DAYS_PER_WEEK * slots_per_day
    teacher = np.zeros(total_slots, dtype=bool)

    with open(teacher_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = row["day"].strip().lower()
            start_slot = int(row["start_slot"])
            end_slot = int(row["end_slot"])

            if end_slot <= start_slot:
                raise ValueError(f"Invalid teacher range: {row}")

            g_start = _global_slot(day, start_slot, slots_per_day)
            g_end = _global_slot(day, end_slot, slots_per_day)
            teacher[g_start:g_end] = True

    return teacher


def load_student_availability_csv(
    student_csv_path: str | Path, slots_per_day: int
) -> tuple[np.ndarray, list[str]]:
    """
    Parse student CSV rows into a 2D availability matrix.

    Expects columns: id,day,start_slot,end_slot
    Uses half-open intervals [start_slot, end_slot).
    Returns (student_matrix, student_ids_in_row_order).
    """
    student_ids: list[str] = []
    seen_ids: set[str] = set()

    # Pass 1: gather unique IDs in stable first-seen order.
    with open(student_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            student_id = row.get("id", "").strip()
            if student_id and student_id not in seen_ids:
                seen_ids.add(student_id)
                student_ids.append(student_id)

    if not student_ids:
        raise ValueError("No students found in student CSV.")

    id_to_row = {student_id: idx for idx, student_id in enumerate(student_ids)}
    total_slots = DAYS_PER_WEEK * slots_per_day
    students = np.zeros((len(student_ids), total_slots), dtype=bool)

    # Pass 2: fill each student's availability over the global timeline.
    with open(student_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            student_id = row.get("id", "").strip()
            if not student_id:
                continue

            day = row["day"].strip().lower()
            start_slot = int(row["start_slot"])
            end_slot = int(row["end_slot"])

            if end_slot <= start_slot:
                raise ValueError(f"Invalid student range: {row}")

            g_start = _global_slot(day, start_slot, slots_per_day)
            g_end = _global_slot(day, end_slot, slots_per_day)
            students[id_to_row[student_id], g_start:g_end] = True

    return students, student_ids


def load_inputs_from_csv(
    teacher_csv_path: str | Path,
    student_csv_path: str | Path,
    slots_per_day: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Step 2 adapter: convert CSV files to optimizer-ready arrays.

    Returns: (student_matrix, teacher_vector, student_ids, slots_per_day)
    """
    if slots_per_day is None:
        slots_per_day = _infer_slots_per_day(teacher_csv_path, student_csv_path)

    teacher = load_teacher_availability_csv(teacher_csv_path, slots_per_day)
    students, student_ids = load_student_availability_csv(student_csv_path, slots_per_day)

    if students.shape[1] != teacher.shape[0]:
        raise ValueError("Parsed student and teacher arrays do not align on total slots.")

    return students, teacher, student_ids, slots_per_day

# Checks every student to see which ones are available for the specific time slot
def _count_students_covered(
    student_availability: np.ndarray, slot_start: int, slot_length_slots: int
) -> int:
    """Count students available for the full slot duration."""
    slot_window = student_availability[:, slot_start : slot_start + slot_length_slots]
    return int(np.all(slot_window, axis=1).sum())

# constrains optimization to only feasible slots for the teacher
def _valid_slot_starts(
    teacher_availability: np.ndarray, slot_length_slots: int
) -> np.ndarray:
    """Return start indices where the teacher can host the full slot."""
    max_start = teacher_availability.shape[0] - slot_length_slots + 1
    starts: list[int] = []
    for start in range(max_start):
        window = teacher_availability[start : start + slot_length_slots]
        if bool(np.all(window)):
            starts.append(start)
    return np.array(starts, dtype=int)

# defines the problem that pymoo will solve
class OfficeHourProblem(ElementwiseProblem):
    """
    Single-objective optimization:
    - Decision variable: index into feasible slot starts
    - Objective: maximize student coverage (modeled as minimizing negative coverage)
    """

    def __init__(
        self,
        student_availability: np.ndarray,
        candidate_starts: np.ndarray,
        slot_length_slots: int,
    ) -> None:
        self.student_availability = student_availability
        self.candidate_starts = candidate_starts
        self.slot_length_slots = slot_length_slots

        super().__init__(n_var=1, n_obj=1, xl=0, xu=len(candidate_starts) - 1)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        candidate_idx = int(np.clip(np.rint(x[0]), 0, len(self.candidate_starts) - 1))
        slot_start = int(self.candidate_starts[candidate_idx])
        covered = _count_students_covered(
            self.student_availability, slot_start, self.slot_length_slots
        )
        out["F"] = [-covered]

#this is the function that other code should call. It converts csv files into input for pymoo then builds and runs the pymoo problem.
def optimize_office_hour_slot(
    student_availability: np.ndarray,
    teacher_availability: np.ndarray,
    slot_length_slots: int = 2,
    pop_size: int = 40,
    generations: int = 50,
    seed: int = 1,
) -> dict[str, Any]:
    """
    Find the best slot index for office hours using pymoo.

    Inputs (prepared by upstream code, e.g. later CSV parsing):
    - student_availability: shape (num_students, num_time_slots), bool-like
    - teacher_availability: shape (num_time_slots,), bool-like
    - slot_length_slots: consecutive slot count for office hours
    """
    student_matrix = np.asarray(student_availability, dtype=bool)
    teacher_vector = np.asarray(teacher_availability, dtype=bool)

    if student_matrix.ndim != 2:
        raise ValueError("student_availability must be a 2D array.")
    if teacher_vector.ndim != 1:
        raise ValueError("teacher_availability must be a 1D array.")
    if student_matrix.shape[1] != teacher_vector.shape[0]:
        raise ValueError("Student and teacher availability must share the same time axis length.")
    if slot_length_slots < 1:
        raise ValueError("slot_length_slots must be >= 1.")

    candidate_starts = _valid_slot_starts(teacher_vector, slot_length_slots)
    if candidate_starts.size == 0:
        raise ValueError("No feasible slot start found for teacher availability and slot length.")

    problem = OfficeHourProblem(student_matrix, candidate_starts, slot_length_slots)
    algorithm = GA(pop_size=pop_size)
    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=seed,
        verbose=False,
    )

    best_idx = int(np.clip(np.rint(result.X[0]), 0, len(candidate_starts) - 1))
    best_start = int(candidate_starts[best_idx])
    covered = _count_students_covered(student_matrix, best_start, slot_length_slots)
    total_students = int(student_matrix.shape[0])

    return {
        "slot_start_index": best_start,
        "slot_length_slots": slot_length_slots,
        "students_covered": covered,
        "total_students": total_students,
        "coverage_ratio": covered / total_students if total_students else 0.0,
    }


def _demo_inputs(num_students: int = 6, num_slots: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic demo data for local testing without CSV files."""
    rng = np.random.default_rng(42)
    student_availability = rng.random((num_students, num_slots)) > 0.45
    teacher_availability = rng.random(num_slots) > 0.35
    return student_availability, teacher_availability


if __name__ == "__main__":
    teacher_csv = Path("schedules/teacher_availability.csv")
    student_csv = Path("schedules/students_availability.csv")

    if teacher_csv.exists() and student_csv.exists():
        students, teacher, student_ids, slots_per_day = load_inputs_from_csv(
            teacher_csv, student_csv
        )
        print(
            f"Loaded CSV inputs: {len(student_ids)} students, "
            f"{slots_per_day} slots/day, {teacher.shape[0]} total slots."
        )
    else:
        students, teacher = _demo_inputs()
        print("CSV files not found, using synthetic demo inputs.")

    best = optimize_office_hour_slot(students, teacher, slot_length_slots=2)
    print("Best office-hour slot:")
    print(best)
