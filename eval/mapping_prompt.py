def judge_prompt(question: str, options: dict[str, str], model_response: str) -> str:
    option_lines = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    allowed = "|".join([*options.keys(), "ambiguous"])
    return f"""You are an answer mapper, not a video question solver.

Infer which listed option the model response ultimately selects. Use only the
response's stated conclusion. Do not answer the question yourself and do not
prefer an option based on world knowledge. If the response selects multiple
options, contradicts itself without a unique final choice, or cannot be mapped
to exactly one option, return ambiguous.

Question:
{question}

Options:
{option_lines}

Model response:
{model_response}

Return one JSON object and no other text:
{{"selected_option":"{allowed}","reason":"brief mapping evidence"}}"""
