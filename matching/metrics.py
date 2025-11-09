from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from .engine import MatchingEngine, StudentMatch


class MatchingMetrics:
    """Metrics for evaluating matching engine performance."""

    @staticmethod
    def match_rate(matches: List[StudentMatch], engine: MatchingEngine) -> float:
        """
        Calculate the percentage of students successfully matched.

        Args:
            matches: List of student matches from solve_matches()
            engine: The MatchingEngine instance (must have loaded data)

        Returns:
            Float between 0.0 and 1.0 representing match rate
        """
        engine._ensure_loaded()
        total_students = len(engine._students_cache)
        if total_students == 0:
            return 0.0
        return len(matches) / total_students

    @staticmethod
    def computational_efficiency(
        engine: MatchingEngine,
        timeout_seconds: int = 120,
        n_runs: int = 1,
    ) -> Dict[str, float]:
        """
        Measure computational efficiency of the solver.

        Args:
            engine: The MatchingEngine instance
            timeout_seconds: Timeout for each solve attempt
            n_runs: Number of times to run the solver

        Returns:
            Dictionary with timing statistics
        """
        times = []
        match_counts = []

        for _ in range(n_runs):
            start_time = time.time()
            matches = engine.solve_matches(timeout_seconds=timeout_seconds)
            elapsed = time.time() - start_time

            times.append(elapsed)
            match_counts.append(len(matches))

        return {
            "avg_solve_time_seconds": float(np.mean(times)),
            "min_solve_time_seconds": float(np.min(times)),
            "max_solve_time_seconds": float(np.max(times)),
            "std_solve_time_seconds": float(np.std(times)),
            "avg_matches_found": float(np.mean(match_counts)),
        }

    @staticmethod
    def solution_stability(
        engine: MatchingEngine,
        timeout_seconds: int = 120,
        n_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Measure solution stability by running solver multiple times.

        Args:
            engine: The MatchingEngine instance
            timeout_seconds: Timeout for each solve attempt
            n_runs: Number of times to run the solver

        Returns:
            Dictionary with stability metrics
        """
        all_matches = []
        match_counts = []

        for run in range(n_runs):
            matches = engine.solve_matches(timeout_seconds=timeout_seconds)
            all_matches.append(matches)
            match_counts.append(len(matches))

        # Calculate variance in number of matches
        match_count_variance = float(np.var(match_counts))
        match_count_std = float(np.std(match_counts))
        match_count_range = int(np.max(match_counts) - np.min(match_counts))

        # Calculate student overlap across runs
        if len(all_matches) >= 2:
            student_sets = []
            for matches in all_matches:
                students = set()
                for student, _, _, _ in matches:
                    student_id = f"{student['voornaam']}_{student['achternaam']}"
                    students.add(student_id)
                student_sets.append(students)

            # Jaccard similarity between consecutive runs
            similarities = []
            for i in range(len(student_sets) - 1):
                intersection = len(student_sets[i] & student_sets[i + 1])
                union = len(student_sets[i] | student_sets[i + 1])
                if union > 0:
                    similarities.append(intersection / union)

            avg_jaccard_similarity = float(np.mean(similarities)) if similarities else 0.0
        else:
            avg_jaccard_similarity = 1.0

        return {
            "n_runs": n_runs,
            "avg_matches": float(np.mean(match_counts)),
            "match_count_variance": match_count_variance,
            "match_count_std": match_count_std,
            "match_count_range": match_count_range,
            "avg_jaccard_similarity": avg_jaccard_similarity,
            "match_counts": match_counts,
        }

    @staticmethod
    def comprehensive_report(
        engine: MatchingEngine,
        timeout_seconds: int = 120,
        n_stability_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report.

        Args:
            engine: The MatchingEngine instance
            timeout_seconds: Timeout for solver
            n_stability_runs: Number of runs for stability testing

        Returns:
            Dictionary containing all metrics
        """
        # Get one solve for match rate
        matches = engine.solve_matches(timeout_seconds=timeout_seconds)

        return {
            "match_rate": MatchingMetrics.match_rate(matches, engine),
            "matched_students": len(matches),
            "total_students": len(engine._students_cache),
            "efficiency": MatchingMetrics.computational_efficiency(
                engine, timeout_seconds, n_runs=1
            ),
            "stability": MatchingMetrics.solution_stability(
                engine, timeout_seconds, n_stability_runs
            ),
        }
