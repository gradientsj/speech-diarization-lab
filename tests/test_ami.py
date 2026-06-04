"""AMI annotation parsing against hand-written NXT XML fragments."""

from diarlab.ami import parse_segments, parse_words

SEGMENTS_XML = b"""<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="X.A.segs" xmlns:nite="http://nite.sourceforge.net/">
   <segment nite:id="X.sync.3" channel="0" transcriber_start="0.0" transcriber_end="1.792">
      <nite:child href="X.A.words.xml#id(X.A.words0)..id(X.A.words3)"/>
   </segment>
   <segment nite:id="X.sync.5" channel="0" transcriber_start="5.5" transcriber_end="9.25"/>
   <segment nite:id="X.sync.7" channel="0" transcriber_start="10.0" transcriber_end="10.0"/>
</nite:root>
"""

WORDS_XML = b"""<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="X.A.words" xmlns:nite="http://nite.sourceforge.net/">
   <w nite:id="X.A.words0" starttime="0.37" endtime="0.95">Hmm</w>
   <w nite:id="X.A.words1" starttime="0.95" endtime="1.53">okay</w>
   <w nite:id="X.A.words2" starttime="1.53" endtime="1.53" punc="true">.</w>
   <vocalsound nite:id="X.A.vocalsound0" starttime="2.0" endtime="2.5" type="laugh"/>
   <w nite:id="X.A.words3">untimed</w>
</nite:root>
"""


def test_parse_segments_keeps_positive_spans_with_speaker():
    turns = parse_segments(SEGMENTS_XML, "A")
    assert turns == [
        {"start": 0.0, "end": 1.792, "speaker": "A"},
        {"start": 5.5, "end": 9.25, "speaker": "A"},
    ]  # the zero-length segment is dropped


def test_parse_words_skips_punctuation_vocalsounds_and_untimed():
    words = parse_words(WORDS_XML)
    assert words == [(0.37, "Hmm"), (0.95, "okay")]
