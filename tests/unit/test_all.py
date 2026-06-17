# tests/unit/test_document_processor.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))
import pytest
from services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


def test_chunk_text_basic(processor):
    text = "Hello world. " * 200
    chunks = processor.chunk_text(text)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c["text"]) <= processor.chunk_size + processor.chunk_overlap + 50


def test_chunk_text_short(processor):
    text = "Short text."
    chunks = processor.chunk_text(text)
    # Too short — should be skipped or returned empty
    assert isinstance(chunks, list)


def test_clean_text(processor):
    dirty = "Hello   \n\n  World  \x01\x02"
    clean = processor._clean_text(dirty)
    assert "\x01" not in clean
    assert "  " not in clean  # collapsed


# tests/unit/test_ats_scorer.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))
from services.resume_analyzer import ResumeAnalyzer, TECH_SKILLS


def test_skill_extraction():
    analyzer = ResumeAnalyzer.__new__(ResumeAnalyzer)
    text = "Experienced Python developer with machine learning and docker skills."
    skills = analyzer._extract_skills(text)
    assert "python" in skills["technical"]
    assert "machine learning" in skills["technical"]
    assert "docker" in skills["technical"]


def test_action_verb_count():
    analyzer = ResumeAnalyzer.__new__(ResumeAnalyzer)
    text = "Developed and deployed a machine learning model. Led a team of 5."
    count = analyzer._count_action_verbs(text)
    assert count >= 2


def test_quantified_achievements():
    analyzer = ResumeAnalyzer.__new__(ResumeAnalyzer)
    assert analyzer._has_quantified_achievements("Increased revenue by 35%") is True
    assert analyzer._has_quantified_achievements("Worked on various projects") is False


def test_ats_score_range():
    analyzer = ResumeAnalyzer.__new__(ResumeAnalyzer)
    analyzer.llm = None
    resume = """
    Software Engineer with 3 years of experience.
    Skills: Python, Machine Learning, Docker, AWS, SQL
    Education: Bachelor of Science in Computer Science, University XYZ
    Experience: Developed ML pipelines. Increased accuracy by 20%.
    Projects: Built REST API using FastAPI. Led team of 4 engineers.
    Summary: Results-driven engineer with expertise in AI.
    """
    skills = analyzer._extract_skills(resume)
    action_count = analyzer._count_action_verbs(resume)
    has_numbers = analyzer._has_quantified_achievements(resume)
    sections = analyzer._check_sections(resume)
    result = analyzer._calculate_ats_score(resume, "", skills, action_count, has_numbers, sections)
    assert 0 <= result["score"] <= 100
    assert "breakdown" in result


# tests/unit/test_vector_store.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))
import pytest


def test_vector_store_search_returns_list():
    """Test that search returns a list (requires sentence-transformers installed)."""
    try:
        from services.vector_store import VectorStore
        store = VectorStore()
        chunks = [
            {"text": "Python is a programming language.", "chunk_index": 0, "word_count": 6, "metadata": {}},
            {"text": "Machine learning uses neural networks.", "chunk_index": 1, "word_count": 5, "metadata": {}},
            {"text": "Natural language processing handles text.", "chunk_index": 2, "word_count": 5, "metadata": {}},
        ]
        store.build_index(chunks)
        results = store.search("programming language", k=2)
        assert isinstance(results, list)
        assert len(results) <= 2
        assert "text" in results[0]
        assert "score" in results[0]
    except ImportError:
        pytest.skip("sentence-transformers not installed")
