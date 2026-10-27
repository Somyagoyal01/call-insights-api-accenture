"""Synthetic health-insurance contact-centre transcripts with labels (development data only).

Each call has an intent, a customer mood and whether it ended in a transfer.
Transfers are more likely when the customer is upset and for intents that
need a specialist (appeals, billing reversals, prior authorisation).
"""
from __future__ import annotations

import random

INTENTS = {
    "claim_status": {
        "ask": ["can you check the status of my claim for {thing}", "has my claim for {thing} been paid",
                "the hospital says my {thing} claim is still pending", "what happened to my claim from last month"],
        "agent": ["The claim is in review and should finish in 10 to 14 days.", "I can see the claim was processed and paid.",
                  "The claim was denied for missing documents from the provider."],
        "base_transfer": 0.25,
    },
    "prior_auth": {
        "ask": ["does {thing} need prior authorization", "my doctor needs approval for {thing}",
                "is pre-authorization required for {thing} on my plan", "the clinic is waiting on authorization for {thing}"],
        "agent": ["Your provider submits the authorization request and you get a letter once approved.",
                  "The authorization is still under clinical review."],
        "base_transfer": 0.35,
    },
    "id_card": {
        "ask": ["I need a new member ID card", "my insurance card is damaged, can I get another",
                "how do I get a digital ID card", "I never received my member card"],
        "agent": ["I've ordered a replacement card, and the digital card is in the app.", "Your new card ships in 5 days."],
        "base_transfer": 0.05,
    },
    "billing_dispute": {
        "ask": ["I was billed twice for {thing}", "why is there a charge for {thing} on my statement",
                "my bill for {thing} looks wrong", "I want to dispute a charge for {thing}"],
        "agent": ["That charge is your copay for the visit.", "I see a duplicate line item for that date."],
        "base_transfer": 0.4,
    },
    "find_provider": {
        "ask": ["can you find an in-network {spec} near me", "which {spec} takes my plan in my area",
                "I need a list of {spec}s covered by my policy", "is there a {spec} nearby that is in network"],
        "agent": ["I found three in-network providers within 5 km and sent you the list.", "Here are in-network options near you."],
        "base_transfer": 0.05,
    },
    "pharmacy_refill": {
        "ask": ["I need a refill of my {drug}", "can I get a 90 day supply of {drug} by mail",
                "the pharmacy says {drug} is not covered", "how do I refill {drug} through mail order"],
        "agent": ["Mail order covers a 90 day supply; I've placed the order.", "That drug moved tiers in the formulary this month."],
        "base_transfer": 0.2,
    },
}
THINGS = ["an MRI", "a blood test", "an ER visit", "physiotherapy", "a CT scan", "cataract surgery", "a specialist visit"]
SPECS = ["cardiologist", "dermatologist", "paediatrician", "orthopaedic doctor", "gynaecologist"]
DRUGS = ["metformin", "insulin", "atorvastatin", "amlodipine", "levothyroxine"]
UPSET = ["this is ridiculous", "I've called three times already", "this is so frustrating", "nobody is helping me",
         "I'm really worried about this"]
CALM = ["thanks", "ok great", "that makes sense, thank you", "perfect", "alright, appreciate it"]


def make_call(rng: random.Random, i: int) -> dict:
    intent = rng.choice(list(INTENTS))
    spec = INTENTS[intent]
    upset = rng.random() < 0.35
    ask = rng.choice(spec["ask"]).format(thing=rng.choice(THINGS), spec=rng.choice(SPECS), drug=rng.choice(DRUGS))
    lines = [f"Customer: {ask}", f"Agent: {rng.choice(spec['agent'])}"]
    if rng.random() < 0.3:
        lines.append(f"Customer: my member number is MBR-{rng.randint(10_000_000, 99_999_999)}")
    p = spec["base_transfer"] + (0.45 if upset else 0.0)
    transferred = int(rng.random() < p)
    lines.append(f"Customer: {rng.choice(UPSET) if upset else rng.choice(CALM)}")
    if transferred:
        lines.append("Agent: Let me transfer you to a specialist team.")
    return {"id": f"C{i:05d}", "intent": intent, "upset": int(upset), "transferred": transferred, "text": "\n".join(lines)}


def corpus(n: int = 1200, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    return [make_call(rng, i) for i in range(n)]
