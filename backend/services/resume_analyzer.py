# backend/services/resume_analyzer.py
"""
Resume Analyzer Service
Features: Skill extraction, ATS scoring, JD matching, gap analysis, cover letter generation
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.llm_factory import get_llm_provider
from typing import Dict, List, Tuple


# ── Skill keywords database ───────────────────────────────────────────
TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "r",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "keras", "pandas", "numpy", "matplotlib",
    "fastapi", "django", "flask", "react", "node.js", "docker", "kubernetes",
    "aws", "gcp", "azure", "git", "linux", "mongodb", "postgresql", "mysql",
    "data analysis", "data science", "ai", "neural networks", "llm",
    "rag", "langchain", "rest api", "microservices", "ci/cd", "devops",
    "spark", "hadoop", "tableau", "power bi", "excel", "matlab",
]

SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving", "critical thinking",
    "time management", "adaptability", "creativity", "collaboration", "presentation",
    "project management", "research", "analytical", "attention to detail",
]

ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "led", "managed", "created",
    "achieved", "improved", "increased", "reduced", "deployed", "optimized",
    "analyzed", "researched", "collaborated", "delivered", "engineered",
    "automated", "integrated", "published", "presented", "mentored",
]

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "b.s.", "m.s.", "b.e.", "m.e.", "bsc", "msc",
    "degree", "university", "college", "institute", "cgpa", "gpa",
]


class ResumeAnalyzer:

    def __init__(self, provider: str = None):
        self.llm = get_llm_provider(provider)

    # ── Main Analysis Entry Point ─────────────────────────────────────

    def analyze(self, resume_text: str, job_description: str = "") -> Dict:
        """Full resume analysis pipeline."""
        skills_found   = self._extract_skills(resume_text)
        education      = self._extract_education(resume_text)
        experience     = self._extract_experience(resume_text)
        action_count   = self._count_action_verbs(resume_text)
        has_numbers    = self._has_quantified_achievements(resume_text)
        sections_found = self._check_sections(resume_text)

        # ATS Score (no JD needed)
        ats_result = self._calculate_ats_score(
            resume_text, job_description, skills_found,
            action_count, has_numbers, sections_found
        )

        # JD-specific analysis
        jd_analysis = {}
        if job_description:
            jd_analysis = self._compare_with_jd(resume_text, job_description, skills_found)

        # LLM-powered suggestions
        suggestions  = self._generate_suggestions(resume_text, job_description, ats_result["score"])
        cover_letter = ""
        if job_description:
            cover_letter = self._generate_cover_letter(resume_text, job_description)

        return {
            "ats_score":        ats_result["score"],
            "ats_breakdown":    ats_result["breakdown"],
            "skills_found":     skills_found,
            "education":        education,
            "experience_years": experience,
            "action_verbs":     action_count,
            "has_numbers":      has_numbers,
            "sections_found":   sections_found,
            "jd_analysis":      jd_analysis,
            "suggestions":      suggestions,
            "cover_letter":     cover_letter,
        }

    # ── ATS Score Calculation ─────────────────────────────────────────

    def _calculate_ats_score(self, resume: str, jd: str, skills: Dict,
                              action_count: int, has_numbers: bool,
                              sections: Dict) -> Dict:
        score = 0
        breakdown = {}

        # Factor 1: Keyword Match with JD (35 pts)
        if jd:
            jd_keywords  = self._extract_keywords(jd)
            res_keywords = self._extract_keywords(resume)
            overlap      = len(set(jd_keywords) & set(res_keywords))
            kw_score     = min(35, round((overlap / max(len(jd_keywords), 1)) * 35, 1))
            breakdown["keyword_match"]   = {"score": kw_score, "max": 35,
                                            "matched": overlap, "total": len(jd_keywords)}
        else:
            kw_score = 17.5   # half marks if no JD
            breakdown["keyword_match"] = {"score": kw_score, "max": 35, "note": "No JD provided"}
        score += kw_score

        # Factor 2: Resume Structure (20 pts)
        section_count = sum(1 for v in sections.values() if v)
        struct_score  = round((section_count / 5) * 20, 1)
        breakdown["structure"] = {"score": struct_score, "max": 20,
                                  "sections_found": section_count, "sections_needed": 5}
        score += struct_score

        # Factor 3: Skills Coverage (20 pts)
        total_skills = len(skills.get("technical", [])) + len(skills.get("soft", []))
        skill_score  = min(20, round(total_skills * 1.5, 1))
        breakdown["skills"] = {"score": skill_score, "max": 20, "total_skills": total_skills}
        score += skill_score

        # Factor 4: Action Verbs (10 pts)
        verb_score = min(10, action_count * 1.5)
        breakdown["action_verbs"] = {"score": verb_score, "max": 10, "verbs_found": action_count}
        score += verb_score

        # Factor 5: Education (10 pts)
        edu_score = 10 if any(kw in resume.lower() for kw in EDUCATION_KEYWORDS) else 3
        breakdown["education"] = {"score": edu_score, "max": 10}
        score += edu_score

        # Factor 6: Quantified Achievements (5 pts)
        num_score = 5 if has_numbers else 0
        breakdown["quantified_achievements"] = {"score": num_score, "max": 5}
        score += num_score

        return {"score": round(min(score, 100), 1), "breakdown": breakdown}

    # ── Skill Extraction ──────────────────────────────────────────────

    def _extract_skills(self, text: str) -> Dict:
        text_lower = text.lower()
        tech  = [s for s in TECH_SKILLS  if s in text_lower]
        soft  = [s for s in SOFT_SKILLS  if s in text_lower]
        return {"technical": list(set(tech)), "soft": list(set(soft))}

    # ── JD Comparison ─────────────────────────────────────────────────

    def _compare_with_jd(self, resume: str, jd: str, skills: Dict) -> Dict:
        jd_lower  = jd.lower()
        res_lower = resume.lower()

        # Required skills from JD
        jd_tech   = [s for s in TECH_SKILLS if s in jd_lower]
        jd_soft   = [s for s in SOFT_SKILLS if s in jd_lower]
        all_jd_skills = jd_tech + jd_soft

        # Missing skills
        all_resume_skills = skills["technical"] + skills["soft"]
        missing = [s for s in all_jd_skills if s not in all_resume_skills]

        # Semantic similarity (simple keyword-based)
        jd_kw  = set(self._extract_keywords(jd))
        res_kw = set(self._extract_keywords(resume))
        overlap = jd_kw & res_kw
        similarity = round(len(overlap) / max(len(jd_kw), 1) * 100, 1)

        return {
            "similarity_percent": similarity,
            "jd_skills_required": all_jd_skills,
            "skills_matched":     [s for s in all_jd_skills if s in all_resume_skills],
            "skills_missing":     missing[:15],
            "keyword_overlap":    list(overlap)[:20],
        }

    # ── LLM: Improvement Suggestions ─────────────────────────────────

    def _generate_suggestions(self, resume: str, jd: str, ats_score: float) -> List[str]:
        system_prompt = (
            "You are an expert career coach and resume reviewer. "
            "Provide actionable, specific, prioritized suggestions."
        )
        jd_section = f"\nJob Description:\n{jd[:1500]}" if jd else ""
        user_prompt = (
            f"Resume:\n{resume[:2500]}{jd_section}\n\n"
            f"Current ATS Score: {ats_score}/100\n\n"
            "Provide exactly 8 specific improvement suggestions as a numbered list. "
            "Each suggestion should be actionable and explain WHY it helps ATS or recruiter appeal. "
            "Focus on: missing keywords, weak action verbs, missing metrics, format issues, skill gaps."
        )
        response = self.llm.generate(system_prompt, user_prompt, max_tokens=1000, temperature=0.4)

        # Parse numbered list
        raw = response.text
        items = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|$)', raw, re.DOTALL)
        if items:
            return [item.strip() for item in items[:8]]
        return [line.strip() for line in raw.split("\n") if line.strip() and len(line.strip()) > 20][:8]

    # ── LLM: Cover Letter Generation ──────────────────────────────────

    def _generate_cover_letter(self, resume: str, jd: str) -> str:
        system_prompt = (
            "You are an expert career counselor specializing in writing compelling, "
            "personalized cover letters that stand out to hiring managers."
        )
        user_prompt = (
            f"Resume Information:\n{resume[:2000]}\n\n"
            f"Job Description:\n{jd[:1500]}\n\n"
            "Write a professional cover letter in business format. Include:\n"
            "- Professional greeting\n"
            "- Opening paragraph: express interest and mention the specific role\n"
            "- Body paragraph 1: highlight 2-3 most relevant experiences/skills\n"
            "- Body paragraph 2: show knowledge of the company/role requirements\n"
            "- Closing paragraph: call to action\n"
            "- Professional sign-off\n"
            "Tone: professional, confident, personalized. Length: 3-4 paragraphs."
        )
        response = self.llm.generate(system_prompt, user_prompt, max_tokens=800, temperature=0.5)
        return response.text

    # ── LLM: Optimized Professional Summary ───────────────────────────

    def generate_optimized_summary(self, resume: str, jd: str = "") -> str:
        system_prompt = "You are an expert resume writer."
        user_prompt = (
            f"Based on this resume:\n{resume[:2000]}\n"
            + (f"\nTarget Job:\n{jd[:800]}\n" if jd else "")
            + "\nWrite an optimized 3-4 sentence professional summary that:\n"
            "1. Starts with years of experience and role title\n"
            "2. Highlights 2-3 strongest technical skills\n"
            "3. Mentions 1-2 key achievements with numbers if possible\n"
            "4. Ends with what value you bring to the employer\n"
            "Make it ATS-friendly with relevant keywords."
        )
        return self.llm.generate(system_prompt, user_prompt, max_tokens=300, temperature=0.4).text

    # ── Helpers ───────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords (3+ chars, not stopwords)."""
        stopwords = {"the", "and", "for", "are", "with", "that", "this", "from",
                     "have", "will", "your", "our", "all", "any", "can", "not",
                     "was", "but", "you", "we", "is", "in", "to", "of", "a", "an"}
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.]*\b', text.lower())
        return [w for w in words if len(w) > 2 and w not in stopwords]

    def _extract_education(self, text: str) -> List[str]:
        patterns = [
            r'\b(B\.?S\.?|B\.?E\.?|BSc|B\.?Tech|M\.?S\.?|M\.?E\.?|MSc|M\.?Tech|PhD|MBA)\b',
            r'\b(bachelor|master|doctorate|diploma)\b',
        ]
        found = []
        for p in patterns:
            found += re.findall(p, text, re.IGNORECASE)
        return list(set(found))

    def _extract_experience(self, text: str) -> str:
        match = re.search(r'(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)', text, re.IGNORECASE)
        return match.group(1) + " years" if match else "Not specified"

    def _count_action_verbs(self, text: str) -> int:
        text_lower = text.lower()
        return sum(1 for v in ACTION_VERBS if v in text_lower)

    def _has_quantified_achievements(self, text: str) -> bool:
        patterns = [r'\d+%', r'\$\d+', r'\d+x\b', r'increased by \d+', r'reduced by \d+']
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _check_sections(self, text: str) -> Dict:
        text_lower = text.lower()
        return {
            "summary":    any(k in text_lower for k in ["summary", "objective", "profile"]),
            "experience": any(k in text_lower for k in ["experience", "work history", "employment"]),
            "education":  any(k in text_lower for k in ["education", "academic", "degree"]),
            "skills":     any(k in text_lower for k in ["skills", "technologies", "expertise"]),
            "projects":   any(k in text_lower for k in ["projects", "portfolio", "achievements"]),
        }
