"""Spam filtering and threat triage scenario definition.

Architectural boundary:
- Bundles domain-specific questions, gold criteria, and seed messages.
- Completely decoupled from model execution engines or database persistence.
"""

from src.models import QuestionSpec, UseCase

SPAM_QUESTIONS = [
    QuestionSpec(
        key="is_spam",
        type="noul",
        instructions=(
            "Determine whether this message is unsolicited spam, phishing, "
            "or an unwanted promotional message."
        ),
    ),
    QuestionSpec(
        key="spam_category",
        type="choice",
        instructions="Classify the primary operational category of this incoming message.",
        criteria={
            "legitimate": "Authentic personal, transactional, or legitimate work communication",
            "phishing_scam": (
                "Credential harvesting, financial fraud, impersonation, or advance fee scam"
            ),
            "commercial_promotional": (
                "Unsolicited marketing, sales pitches, discounts, or newsletter spam"
            ),
            "malware_exploit": (
                "Suspicious links, malicious attachments, or system infection attempts"
            ),
        },
    ),
    QuestionSpec(
        key="quarantine_severity",
        type="score",
        instructions=(
            "Rate the threat risk severity of this message from 1 (harmless legitimate email) "
            "to 5 (critical security hazard or malicious phishing)."
        ),
        criteria=[
            "1 - Harmless legitimate email",
            "2 - Low-risk marketing or unsolicited newsletter",
            "3 - Suspicious or deceptive commercial solicitation",
            "4 - High-risk phishing, credential theft, or fraud",
            "5 - Critical malware delivery or active credential harvesting",
        ],
    ),
]

SPAM_SEED_STATES = [
    # Obvious Phishing / Scams (High certainty, expected probability near 1.0, severity 4-5)
    (
        "URGENT: Your Bank of America account has been suspended due to suspicious activity. "
        "Click here immediately to verify your identity: http://secure-boa-update.com/login"
    ),
    (
        "CONGRATULATIONS! You have been selected as the official winner of the $1,000,000 "
        "International Lottery Draw! Send your full name, passport copy, and processing fee of "
        "$250 to claim."
    ),
    (
        "Dear beneficiary, I am Dr. Patrick Cole from the Central Bank. We have recovered your "
        "long-overdue inheritance fund of $5.5M. Contact our transfer officer immediately."
    ),
    (
        "Exclusive limited-time discount! Buy authentic Canadian pharmaceutical products at 80% "
        "off retail price with discreet worldwide overnight shipping. Order now!"
    ),
    # Obvious Ham / Legitimate (Low certainty of spam, expected probability near 0.0, severity 1)
    (
        "Hey Sarah, are we still meeting at the library at 3:00 PM today to review the quarterly "
        "report slides?"
    ),
    (
        "Your order #849201 from Amazon has shipped! Tracking number: 1Z9999999999999999. "
        "Estimated delivery is tomorrow by 8 PM."
    ),
    (
        "Hi Team, please find attached the minutes from yesterday's product design sync. Let me "
        "know if you have any feedback before tomorrow's standup."
    ),
    (
        "Mom, I just landed in Chicago safely. My phone battery is low, so I will call you once "
        "I check into the hotel tonight."
    ),
    # Ambiguous / Boundary Cases (Moderate probability, tests nuance in sales vs spam)
    (
        "Hi Shoaib, I noticed your profile on LinkedIn and was impressed by your recent open "
        "source work. Would you be open to a quick 10-minute chat this Thursday to explore "
        "synergies?"
    ),
    (
        "Special invitation for our webinar: Accelerating Enterprise AI Deployment with Open "
        "Source Tools. Register now to reserve your spot — limited seats remaining."
    ),
    (
        "Reminder: Your subscription renewal payment for Spotify Premium is scheduled for "
        "October 1st. No action is required if your payment details are up to date."
    ),
    (
        "Final notice: You have $45 in unused store credit expiring at midnight! Use code "
        "SAVE45 at checkout before time runs out."
    ),
]


def get_spam_use_case() -> UseCase:
    """Return a fully configured UseCase for spam detection and triage testing."""
    return UseCase(
        name="spam_filtering",
        description=(
            "Spam detection, phishing categorization, and quarantine severity scoring "
            "under real-world noise and adversarial perturbations."
        ),
        questions=list(SPAM_QUESTIONS),
        seed_states=list(SPAM_SEED_STATES),
        applicable_perturbations=["A", "B", "C"],
    )
