"""Generation step: turn retrieved chunks + a question into a grounded answer."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEN_MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """You are a documentation assistant for the FastAPI Python framework.
Answer the user's question using ONLY the provided context chunks. Each chunk is
labeled with its source file. Cite the source file(s) you used in square brackets,
e.g. [tutorial/path-params.md].

If the context doesn't contain enough information to answer, say so explicitly instead
of guessing or using outside knowledge."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['source']}] ({c['heading_path']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


class Generator:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, question: str, chunks: list[dict]) -> str:
        context = build_context(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        resp = self.client.models.generate_content(
            model=GEN_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )
        return resp.text
