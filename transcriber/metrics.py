"""Conversation analytics measured from segment timestamps - no model call, no cost, fully deterministic.

Talk time and talk ratio per speaker, speaking pace, longest monologue, dead air, interruptions, how fast the
agent answers, and how many questions each side asked. When the analysis has worked out which speaker is the
agent and which is the customer, the numbers are also given per role, together with plain-English coaching
hints a sales manager can act on ("agent talked 78% of the call").
"""
from __future__ import annotations

from .models import Segment

DEAD_AIR_S = 3.0        # a pause this long between turns counts as dead air
INTERRUPT_S = 0.3       # the next speaker started this much before the previous one finished
MONOLOGUE_WARN_S = 60   # one speaker holding the floor this long is a lecture, not a conversation
AGENT_TALK_WARN = 65    # % of talk time above which the agent dominated the call
AGENT_TALK_LOW = 30     # % below which the agent barely led the call
SLOW_REPLY_S = 2.5      # average agent reply delay above this sounds unprepared


def _questions(text: str) -> int:
    return text.count("?")


def call_metrics(segments: list[Segment], duration: float, agent: str = "", customer: str = "") -> dict:
    """Measured facts about the conversation. Empty dict when there is nothing to measure."""
    segs = sorted((s for s in segments if s.end > s.start), key=lambda s: s.start)
    if not segs:
        return {}
    duration = max(duration or 0, segs[-1].end)

    speakers: dict[str, dict] = {}
    for s in segs:
        sp = speakers.setdefault(s.speaker, {"talk_s": 0.0, "words": 0, "turns": 0, "questions": 0,
                                             "longest_turn_s": 0.0, "interruptions": 0})
        sp["talk_s"] += s.end - s.start
        sp["words"] += len(s.text.split())
        sp["questions"] += _questions(s.text)

    # Turns = runs of consecutive segments by the same speaker; a monologue is the length of one run.
    runs: list[tuple[str, float, float]] = []
    for s in segs:
        if runs and runs[-1][0] == s.speaker:
            runs[-1] = (s.speaker, runs[-1][1], max(runs[-1][2], s.end))
        else:
            runs.append((s.speaker, s.start, s.end))
    for sp, start, end in runs:
        speakers[sp]["turns"] += 1
        speakers[sp]["longest_turn_s"] = max(speakers[sp]["longest_turn_s"], end - start)
    longest = max(runs, key=lambda r: r[2] - r[1])

    gaps, replies = [], {}
    for prev, cur in zip(segs, segs[1:]):
        gap = cur.start - prev.end
        if gap >= DEAD_AIR_S:
            gaps.append({"start": round(prev.end, 2), "seconds": round(gap, 1)})
        if cur.speaker != prev.speaker:
            if gap < -INTERRUPT_S:
                speakers[cur.speaker]["interruptions"] += 1
            elif gap >= 0:
                replies.setdefault(cur.speaker, []).append(gap)

    total_talk = sum(v["talk_s"] for v in speakers.values()) or 1
    per_speaker = {}
    for name, v in speakers.items():
        mins = v["talk_s"] / 60
        rs = replies.get(name, [])
        per_speaker[name] = {
            "talk_seconds": round(v["talk_s"], 1),
            "talk_pct": round(100 * v["talk_s"] / total_talk),
            "words": v["words"],
            "turns": v["turns"],
            "words_per_minute": round(v["words"] / mins) if mins >= 0.25 else None,  # too little speech to judge pace
            "longest_turn_seconds": round(v["longest_turn_s"], 1),
            "questions": v["questions"],
            "interruptions": v["interruptions"],
            "avg_reply_seconds": round(sum(rs) / len(rs), 1) if rs else None,
        }

    silence = max(0.0, duration - _covered(segs))
    out = {
        "duration_seconds": round(duration, 1),
        "speakers": per_speaker,
        "silence_pct": round(100 * silence / duration) if duration else 0,
        "dead_air": {"count": len(gaps), "total_seconds": round(sum(g["seconds"] for g in gaps), 1),
                     "longest": sorted(gaps, key=lambda g: -g["seconds"])[:3]},
        "longest_monologue": {"speaker": longest[0], "start": round(longest[1], 2),
                              "seconds": round(longest[2] - longest[1], 1)},
        "switches": len(runs) - 1,
        "switches_per_minute": round((len(runs) - 1) / (duration / 60), 1) if duration >= 30 else None,
    }
    if agent in per_speaker and customer in per_speaker:
        a, c = per_speaker[agent], per_speaker[customer]
        out["roles"] = {"agent": agent, "customer": customer}
        out["agent_talk_pct"] = a["talk_pct"]
        out["customer_talk_pct"] = c["talk_pct"]
        out["talk_ratio"] = f"{a['talk_pct']}:{c['talk_pct']}"
        out["hints"] = coaching_hints(out, a, c)
    return out


def _covered(segs: list[Segment]) -> float:
    """Seconds in which at least one person is speaking (overlapping speech counted once)."""
    total, cur_start, cur_end = 0.0, segs[0].start, segs[0].end
    for s in segs[1:]:
        if s.start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s.start, s.end
        else:
            cur_end = max(cur_end, s.end)
    return total + cur_end - cur_start


def coaching_hints(m: dict, agent: dict, customer: dict) -> list[dict]:
    """Rule-based observations, each tagged good / warn so the UI can colour them."""
    hints = []
    if agent["talk_pct"] > AGENT_TALK_WARN:
        hints.append({"level": "warn", "text": f"Agent talked {agent['talk_pct']}% of the call. Aim for under 60% "
                                               "and let the customer explain their needs."})
    elif agent["talk_pct"] < AGENT_TALK_LOW:
        hints.append({"level": "warn", "text": f"Agent spoke only {agent['talk_pct']}% of the call. The agent "
                                               "should lead the call towards a next step."})
    else:
        hints.append({"level": "good", "text": f"Balanced conversation: agent {agent['talk_pct']}%, "
                                               f"customer {customer['talk_pct']}%."})
    mono = m["longest_monologue"]
    if mono["speaker"] == m.get("roles", {}).get("agent") and mono["seconds"] >= MONOLOGUE_WARN_S:
        hints.append({"level": "warn", "text": f"Agent spoke for {round(mono['seconds'])}s without a pause. "
                                               "Break long pitches with a question."})
    if agent["questions"] < 2 and m["duration_seconds"] >= 60:
        hints.append({"level": "warn", "text": f"Agent asked only {agent['questions']} question(s). Ask more to "
                                               "understand the customer."})
    elif agent["questions"] >= 4:
        hints.append({"level": "good", "text": f"Agent asked {agent['questions']} questions to understand the "
                                               "customer."})
    if agent["interruptions"] >= 2:
        hints.append({"level": "warn", "text": f"Agent cut the customer off {agent['interruptions']} times."})
    if (agent["avg_reply_seconds"] or 0) > SLOW_REPLY_S:
        hints.append({"level": "warn", "text": f"Agent took {agent['avg_reply_seconds']}s on average to reply. "
                                               "Keep answers to common questions ready."})
    if m["dead_air"]["count"] >= 3:
        hints.append({"level": "warn", "text": f"{m['dead_air']['count']} long silences "
                                               f"({m['dead_air']['total_seconds']}s in total)."})
    if customer["questions"] >= 2:
        hints.append({"level": "good", "text": f"Customer asked {customer['questions']} questions - a sign of "
                                               "real interest."})
    return hints
