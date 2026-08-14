from src.preprocessing import clean_text, normalise_record


def test_clean_text_repairs_contractions():
    assert clean_text("She doesn' t know.") == "She doesn't know."


def test_four_choices_are_required():
    record = {
        "question_text": "Why?",
        "answer_choice_texts": ["one"],
        "rationale_choice_texts": ["a", "b", "c", "d"],
    }
    try:
        normalise_record(record)
        assert False
    except ValueError:
        assert True
