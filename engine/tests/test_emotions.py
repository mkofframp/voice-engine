from voice_engine.emotions import EMOTION_PROMPT, EMOTION_TAGS, Sentence, TagParser


def test_single_sentence_in_chunks():
    p = TagParser()
    assert p.feed("[hap") == []
    assert p.feed("py] Hi the") == []
    assert p.feed("re! ") == [Sentence("happy", "Hi there!")]


def test_two_sentences_two_tags():
    p = TagParser()
    out = p.feed("[curious] What is it? [excited] Wow!")
    out += p.flush()
    assert out == [Sentence("curious", "What is it?"), Sentence("excited", "Wow!")]


def test_untagged_inherits():
    p = TagParser()
    out = p.feed("[sad] Oh no. That hurts.") + p.flush()
    assert out == [Sentence("sad", "Oh no."), Sentence("sad", "That hurts.")]


def test_untagged_start_is_neutral():
    p = TagParser()
    assert p.feed("Hello. ") == [Sentence("neutral", "Hello.")]


def test_decimal_not_sentence_end():
    p = TagParser()
    out = p.feed("[neutral] It is 3.5 meters.") + p.flush()
    assert out == [Sentence("neutral", "It is 3.5 meters.")]


def test_flush_partial():
    p = TagParser()
    assert p.feed("[gentle] okay") == []
    assert p.flush() == [Sentence("gentle", "okay")]


def test_newline_ends_sentence_and_unknown_tag_kept():
    p = TagParser()
    out = p.feed("[zany] No stop\n[happy] Yes") + p.flush()
    assert out == [Sentence("zany", "No stop"), Sentence("happy", "Yes")]


def test_reset():
    p = TagParser()
    p.feed("[sad] hmm")
    p.reset()
    assert p.feed("Hi. ") == [Sentence("neutral", "Hi.")]


def test_prompt_lists_tags():
    assert all(t in EMOTION_PROMPT for t in EMOTION_TAGS)
