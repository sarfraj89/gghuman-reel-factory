"""Beat timing derived from text length, so every video paces itself correctly
without a human editor. Audio cues are generated from the same timeline, which
is why sound and picture never drift apart."""

TYPE_CPS = 38        # chat typing speed, characters / second
DECODE_CPS = 40      # inner-monologue decode speed
READ_WPS = 3.4       # comfortable on-screen reading speed, words / second


def read_time(text, factor=1.0):
    return max(0.9, len(text.split()) / READ_WPS * factor)


def build(spec):
    tl = {}
    tl["hook"] = 0.1
    tl["scene_in"] = 0.35
    tl["you"] = 2.0
    tl["you_end"] = tl["you"] + len(spec["you_line"]) / TYPE_CPS
    tl["said"] = tl["you_end"] + read_time(spec["you_line"], 0.45) + 0.25
    tl["said_end"] = tl["said"] + len(spec["ai_said"]) / TYPE_CPS
    tl["thought"] = tl["said_end"] + read_time(spec["ai_said"], 0.5) + 0.2
    tl["thought_end"] = tl["thought"] + 0.35 + len(spec["ai_thought"]) / DECODE_CPS
    tl["event"] = tl["thought_end"] + read_time(spec["ai_thought"], 0.6)
    tl["twist"] = tl["event"] + 1.5
    tl["end"] = tl["twist"] + max(1.9, read_time(spec["twist"]) + 0.9)
    tl["cta"] = [tl["end"] + 0.25 + i * 0.42 for i in range(3)]
    tl["url"] = tl["cta"][-1] + 0.5
    tl["total"] = tl["url"] + 2.2
    return tl
