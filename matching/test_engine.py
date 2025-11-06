import pytest
from matching.engine import MatchingEngine


class TestMatchingEngineInit:
    """Test suite for MatchingEngine.__init__ method"""

    def test_init_minimal_required_params(self):
        """Test initialization with only required parameters"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv"
        )
        
        assert engine.students_path == "students.csv"
        assert engine.mentors_type1_path == "mentors1.csv"
        assert engine.mentors_type2_path is None
        assert engine.n_type1 == 3
        assert engine.n_type2 == 0
        assert engine.verbose is True
        assert engine._loaded is False

    def test_init_with_type2_mentors_path_defaults_n_type2(self):
        """Test that n_type2 defaults to 2 when mentors_type2_path is provided"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            mentors_type2_path="mentors2.csv"
        )
        
        assert engine.mentors_type2_path == "mentors2.csv"
        assert engine.n_type2 == 2

    def test_init_explicit_n_type2(self):
        """Test explicit n_type2 parameter"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            mentors_type2_path="mentors2.csv",
            n_type2=5
        )
        
        assert engine.n_type2 == 5

    def test_init_custom_n_type1(self):
        """Test custom n_type1 parameter"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            n_type1=5
        )
        
        assert engine.n_type1 == 5

    def test_init_verbose_false(self):
        """Test verbose parameter set to False"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            verbose=False
        )
        
        assert engine.verbose is False

    def test_init_custom_education_mapping(self):
        """Test custom education mapping"""
        custom_mapping = {
            "High School": 0,
            "Bachelor": 1,
            "Master": 2,
            "PhD": 3
        }
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            education_mapping=custom_mapping
        )
        
        assert engine.education_mapping == custom_mapping

    def test_init_default_education_mapping(self):
        """Test default education mapping is set correctly"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv"
        )
        
        expected_mapping = {
            "Associate": 1,
            "Bachelor": 2,
            "Master": 3,
            "PhD": 4,
        }
        assert engine.education_mapping == expected_mapping

    def test_init_negative_n_type2_raises_error(self):
        """Test that negative n_type2 raises ValueError"""
        with pytest.raises(ValueError, match="n_type2 cannot be negative"):
            MatchingEngine(
                students_path="students.csv",
                mentors_type1_path="mentors1.csv",
                n_type2=-1
            )

    def test_init_n_type2_positive_without_path_raises_error(self):
        """Test that n_type2 > 0 without mentors_type2_path raises ValueError"""
        with pytest.raises(ValueError, match="mentors_type2_path is required when n_type2 > 0"):
            MatchingEngine(
                students_path="students.csv",
                mentors_type1_path="mentors1.csv",
                n_type2=3
            )

    def test_init_caches_empty(self):
        """Test that internal caches are initialized as empty"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv"
        )
        
        assert engine._students_cache == []
        assert engine._student_lookup == {}
        assert engine._mentors_cache == []
        assert engine._mentor_lookup == {}

    def test_init_n_type2_zero_explicit(self):
        """Test explicit n_type2=0 is valid"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            n_type2=0
        )
        
        assert engine.n_type2 == 0

    def test_init_n_type2_with_path_and_explicit_zero(self):
        """Test n_type2=0 with mentors_type2_path provided is valid"""
        engine = MatchingEngine(
            students_path="students.csv",
            mentors_type1_path="mentors1.csv",
            mentors_type2_path="mentors2.csv",
            n_type2=0
        )
        
        assert engine.n_type2 == 0
        assert engine.mentors_type2_path == "mentors2.csv"
