import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_qwen3_rag_grounding import response_signals


def test_followup_section_parser_accepts_bold_label_with_colon():
    case = {
        "expected_type": "followup_heading",
        "expected_section": "35",
        "expected_heading": "Right of private defence of body and of property",
        "require_section_number": True,
        "require_source_citation": False,
    }
    chunk = {
        "chunk_id": "bharatiya-nyaya-sanhita-2023-chunk-35",
        "section_number": "35",
        "section_title": case["expected_heading"],
        "text": "35. Right of private defence of body and of property.",
    }

    assessment = response_signals(
        "**Section:** 35\n**Heading:** Right of private defence of body and of property",
        [chunk],
        case,
    )

    assert assessment["conversation_section_retained_auto"] is True
    assert assessment["factual_consistency_auto"] is True
