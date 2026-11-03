#!/usr/bin/env python3
"""List the Gemini models GOOGLE_API_KEY can call, for choosing GEMINI_MODEL."""

import os
import sys

from google import genai


def main() -> int:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("GOOGLE_API_KEY is not set.", file=sys.stderr)
        return 1
    for model in genai.Client(api_key=api_key).models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name.removeprefix("models/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
