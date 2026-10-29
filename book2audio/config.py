DEFAULT_MODEL = "preslaff/omnivoice-bg"
DEFAULT_SR = 24000

# Pauses (seconds)
PAUSE_CHUNK = 0.30       # between chunks inside one paragraph
PAUSE_PARAGRAPH = 0.70   # between paragraphs
PAUSE_SCENE = 1.40       # scene breaks such as "* * *"
CHAPTER_LEAD = 0.8
CHAPTER_TAIL = 2.0

# Spoken once to create a voice when no sample is given; reused as ref_text.
DESIGN_SENTENCE = ("Добре дошли. С този глас ще ви прочета цялата книга – "
                   "спокойно, ясно и глава по глава.")

CHARS_PER_SEC = 13.0     # rough narration speed, used for sanity checks / estimates
DEFAULT_DESIGN_MODEL = "k2-fsa/OmniVoice"   # designs the voice when no sample is given
DEFAULT_PORT = 8765                         # book2audio-server
