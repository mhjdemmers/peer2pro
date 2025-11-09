from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

import pandas as pd
from clingo import Control

StudentMatch = Tuple[Dict[str, str], List[Dict[str, str]], List[Dict[str, str]], str]


class MatchingEngine:
    def __init__(
        self,
        *,
        students_df: pd.DataFrame,
        mentors_type1_df: pd.DataFrame,
        mentors_type2_df: Optional[pd.DataFrame] = None,
        n_type1: int = 3,
        n_type2: Optional[int] = None,
        education_mapping: Optional[Dict[str, int]] = None,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self._students_df = students_df.copy()
        self._mentors_type1_df = mentors_type1_df.copy()
        self._mentors_type2_df = mentors_type2_df.copy() if mentors_type2_df is not None else None
        self.education_mapping = education_mapping or {
            "Associate": 1,
            "Bachelor": 2,
            "Master": 3,
            "PhD": 4,
        }

        self.n_type1 = n_type1
        if self._mentors_type2_df is None and n_type2 > 0:
            if self.verbose:
                print("No Type 2 mentors provided; overriding n_type2 to 0.")
            self.n_type2 = 0
        else:
            self.n_type2 = n_type2

        self._students_cache: List[Dict] = []
        self._student_lookup: Dict[str, Dict] = {}
        self._mentors_cache: List[Dict] = []
        self._mentor_lookup: Dict[str, Dict] = {}
        self._loaded = False

    def load_data(self) -> None:
        students_df = self._students_df
        mentors_type1_df = self._mentors_type1_df
        mentors_type2_df = self._mentors_type2_df

        self._students_cache, self._student_lookup = self._build_students_cache(students_df)
        self._mentors_cache, self._mentor_lookup = self._build_mentors_cache(
            mentors_type1_df, mentors_type2_df
        )
        self._loaded = True

    def solve_matches(self, timeout_seconds: int = 120) -> List[StudentMatch]:
        self._ensure_loaded()

        ctl = Control()
        ctl.add("base", [], self._build_asp_program())
        ctl.ground([("base", [])])

        matches: List[StudentMatch] = []
        best_cost: Optional[Tuple[int, ...]] = None

        def collect_matches(model) -> None:
            nonlocal best_cost

            grouped: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
            for symbol in model.symbols(shown=True):
                if symbol.name != "match":
                    continue

                student_id = symbol.arguments[0].name
                mentor_id = symbol.arguments[1].name
                day = symbol.arguments[2].name

                key = (student_id, day)
                grouped.setdefault(key, {"type1": [], "type2": []})
                mentor_type = self._mentor_lookup[mentor_id]["mentor_type"]
                grouped[key][mentor_type].append(mentor_id)

            temp_matches: List[StudentMatch] = []
            for (student_id, day), mentor_groups in grouped.items():
                student_entry = self._student_lookup[student_id]
                student_data = student_entry["data"].copy()

                mentors_type1 = [
                    self._mentor_lookup[m]["data"].copy() for m in sorted(mentor_groups["type1"])
                ]
                mentors_type2 = [
                    self._mentor_lookup[m]["data"].copy() for m in sorted(mentor_groups["type2"])
                ]

                temp_matches.append((student_data, mentors_type1, mentors_type2, day))

            current_cost = tuple(model.cost or [])
            if best_cost is None or current_cost < best_cost:
                best_cost = current_cost
                matches.clear()
                matches.extend(temp_matches)
                if self.verbose:
                    print(f"Found solution with {len(matches)} matched students (cost: {current_cost})")

        if self.verbose:
            print(f"Solving with {timeout_seconds}s timeout (best model only)...")

        with ctl.solve(on_model=collect_matches, async_=True) as handle:
            handle.wait(timeout_seconds)
            handle.cancel()
            result = handle.get()

        if self.verbose:
            print(f"Result: {result}")
            if result.interrupted:
                print(f"Timeout - returning {len(matches)} students")
            elif result.unsatisfiable:
                print("UNSAT - no valid solution")
            elif result.exhausted:
                print(f"Optimal solution with {len(matches)} students")
            else:
                print(f"Solution with {len(matches)} students")

        return matches

    def export_matches(self, matches: List[StudentMatch], filename: str) -> Optional[pd.DataFrame]:
        if not matches:
            if self.verbose:
                print("No matches to export")
            return None

        rows = []
        for student, mentors_type1, mentors_type2, day in matches:
            rows.append(
                {
                    "Student": f"{student['voornaam']} {student['achternaam']}",
                    "Day": day.capitalize(),
                    "Mentors_Type1": "; ".join(
                        f"{m['voornaam']} {m['achternaam']}" for m in mentors_type1
                    ),
                    "Mentors_Type2": "; ".join(
                        f"{m['voornaam']} {m['achternaam']}" for m in mentors_type2
                    ),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        if self.verbose:
            print(f"Exported {len(matches)} matches to {filename}")
        return df

    def matches_to_dataframe(self, matches: List[StudentMatch]) -> pd.DataFrame:
        rows = []
        for student, mentors_type1, mentors_type2, day in matches:
            rows.append(
                {
                    "student_first_name": student["voornaam"],
                    "student_last_name": student["achternaam"],
                    "day": day,
                    "type1_mentors": [f"{m['voornaam']} {m['achternaam']}" for m in mentors_type1],
                    "type2_mentors": [f"{m['voornaam']} {m['achternaam']}" for m in mentors_type2],
                }
            )
        return pd.DataFrame(rows)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_data()

    def _build_students_cache(self, df: pd.DataFrame):
        cache: List[Dict] = []
        lookup: Dict[str, Dict] = {}

        for idx, row in df.iterrows():
            student_id = f"s{idx}"
            subject_atom = self._normalize_token(row["Onderwerp"])
            availability = [day.lower() for day in self._parse_literal_list(row["Beschikbaarheid"])]

            entry = {
                "id": student_id,
                "education_level": self.education_mapping[row["Opleidingsniveau"]],
                "subject_atom": subject_atom,
                "availability": availability,
                "data": {
                    "voornaam": row["Voornaam"],
                    "achternaam": row["Achternaam"],
                    "opleidingsniveau": row["Opleidingsniveau"],
                    "onderwerp": row["Onderwerp"],
                },
            }

            cache.append(entry)
            lookup[student_id] = entry

        return cache, lookup

    def _build_mentors_cache(
        self, mentors_df: pd.DataFrame, mentors2_df: Optional[pd.DataFrame]
    ):
        cache: List[Dict] = []
        lookup: Dict[str, Dict] = {}

        for idx, row in mentors_df.iterrows():
            mentor_id = f"m{idx}"
            cache_entry = self._mentor_entry(row, mentor_id, "type1", "Type 1")
            cache.append(cache_entry)
            lookup[mentor_id] = cache_entry

        if mentors2_df is not None:
            for idx, row in mentors2_df.iterrows():
                mentor_id = f"m2_{idx}"
                cache_entry = self._mentor_entry(row, mentor_id, "type2", "Type 2")
                cache.append(cache_entry)
                lookup[mentor_id] = cache_entry

        return cache, lookup

    def _mentor_entry(self, row, mentor_id: str, mentor_type: str, label: str) -> Dict:
        subjects = [self._normalize_token(subject) for subject in self._parse_literal_list(row["Onderwerpen"])]
        availability = [day.lower() for day in self._parse_literal_list(row["Beschikbaarheid"])]
        max_students = int(row["Max_Studenten"])

        return {
            "id": mentor_id,
            "mentor_type": mentor_type,
            "education_level": self.education_mapping[row["Opleidingsniveau"]],
            "subjects": subjects,
            "availability": availability,
            "max_students": max_students,
            "data": {
                "voornaam": row["Voornaam"],
                "achternaam": row["Achternaam"],
                "opleidingsniveau": row["Opleidingsniveau"],
                "type": label,
            },
        }

    def _build_asp_program(self) -> str:
        facts = self._generate_asp_facts()
        type2_block = ""
        if self.n_type2 > 0:
            type2_block = f"""
:- match_day(S, Day), #count {{ M : candidate(S,M,Day), mentor_type(M,type2) }} < {self.n_type2}.
{self.n_type2} {{ match(S, M, Day) : candidate(S,M,Day), mentor_type(M,type2) }} {self.n_type2} :- match_day(S, Day).

"""

        return f"""
% Facts from Python
{facts}

% Candidate mentor-student-day triples
candidate(S, M, Day) :-
    student(S),
    mentor(M),
    expertise(S, Subj),
    expertise(M, Subj),
    education(S, ES),
    education(M, EM),
    EM > ES,
    availability(S, Day),
    availability(M, Day).

% Choose which students to match (0 or 1 day per student)
{{ selected(S) }} :- student(S).

% If selected, choose exactly one day
1 {{ match_day(S, Day) : day(Day), availability(S, Day) }} 1 :- selected(S).

% Only allow days where student has enough candidates
:- match_day(S, Day), #count {{ M : candidate(S,M,Day), mentor_type(M,type1) }} < {self.n_type1}.
{type2_block}% Choose exact mentors per type
{self.n_type1} {{ match(S, M, Day) : candidate(S,M,Day), mentor_type(M,type1) }} {self.n_type1} :- match_day(S, Day).

% Respect mentor capacities
:- mentor(M), max_students(M, Max), #count {{ S, Day : match(S, M, Day) }} > Max.

% Maximize selected students
#maximize {{ 1,S : selected(S) }}.

#show match/3.
"""

    def _generate_asp_facts(self) -> str:
        facts: List[str] = []
        days = set()

        for student in self._students_cache:
            facts.append(f"student({student['id']}).")
            facts.append(f"education({student['id']}, {student['education_level']}).")
            facts.append(f"expertise({student['id']}, {student['subject_atom']}).")
            for day in student["availability"]:
                facts.append(f"availability({student['id']}, {day}).")
                days.add(day)

        for mentor in self._mentors_cache:
            facts.append(f"mentor({mentor['id']}).")
            facts.append(f"mentor_type({mentor['id']}, {mentor['mentor_type']}).")
            facts.append(f"education({mentor['id']}, {mentor['education_level']}).")
            for subject in mentor["subjects"]:
                facts.append(f"expertise({mentor['id']}, {subject}).")
            facts.append(f"max_students({mentor['id']}, {mentor['max_students']}).")
            for day in mentor["availability"]:
                facts.append(f"availability({mentor['id']}, {day}).")
                days.add(day)

        for day in sorted(days):
            facts.append(f"day({day}).")

        return "\n".join(facts)

    @staticmethod
    def _parse_literal_list(raw) -> List[str]:
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return []
        else:
            parsed = raw
        return list(parsed) if isinstance(parsed, (list, tuple)) else []

    @staticmethod
    def _normalize_token(text) -> str:
        return text.lower().replace(" ", "_") if isinstance(text, str) else ""