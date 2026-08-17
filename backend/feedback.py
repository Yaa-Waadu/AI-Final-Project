import os

from groq import Groq

FEEDBACK_MODEL = "openai/gpt-oss-120b"

_client = None


def _get_client():
    global _client

    if _client is not None:
        return _client

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    _client = Groq(api_key=api_key)

    return _client


def generate_asl_feedback(predicted_letter, confidence, top3, attempted_letter=None):
    """
    Turns a single ASLCNN prediction into short, encouraging coaching
    feedback for a learner practicing ASL fingerspelling, using Llama 3.3
    70B via Groq. Returns None (rather than raising) if no API key is
    configured or the call fails, so feedback is always best-effort and
    never breaks a prediction.
    """

    client = _get_client()

    if client is None:
        return None

    top3_str = ", ".join(f"{letter.upper()} ({pct:.1f}%)" for letter, pct in top3)

    system_prompt = (
        "You are a supportive, knowledgeable ASL fingerspelling coach helping a "
        "beginner practice the static alphabet. You never invent anatomical or "
        "linguistic facts you are not confident about. You are encouraging but "
        "specific and concise (2-4 sentences, no more)."
    )

    if attempted_letter:
        correct = attempted_letter.lower() == predicted_letter.lower()

        user_prompt = f"""
A learner was attempting to sign the letter '{attempted_letter.upper()}'.
An image classifier predicted '{predicted_letter.upper()}' with {confidence:.1f}% confidence.
Its top-3 predictions were: {top3_str}.
The prediction was {"CORRECT" if correct else "INCORRECT"}.

Write short coaching feedback for the learner:
- If correct and confidence is high, briefly affirm what likely went right.
- If correct but confidence is only moderate, gently note the sign could be cleaner.
- If incorrect, name the likely hand-shape confusion suggested by the top-3 list
  (e.g. finger spacing, thumb position) without overstating certainty, and give
  ONE concrete adjustment to try next.
Keep it warm, brief, and specific to fingerspelling practice.
"""
    else:
        user_prompt = f"""
A learner uploaded a photo of a static ASL hand shape, without specifying which
letter they intended. An image classifier predicted '{predicted_letter.upper()}'
with {confidence:.1f}% confidence. Its top-3 predictions were: {top3_str}.

Write a short, encouraging note confirming what sign was recognized. If
confidence is only moderate, add one concrete tip for making the sign clearer.
Keep it warm, brief, and specific to fingerspelling practice.
"""

    try:
        response = client.chat.completions.create(
            model=FEEDBACK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=300,
            reasoning_effort="low",
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq feedback generation failed: {e}")
        return None
