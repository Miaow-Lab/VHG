from __future__ import annotations


INTEGRATION_COLD_SFT_SYSTEM_PROMPT = """You are an expert calculus problem designer generating indefinite-integration training data.

Generate one polished indefinite-integration candidate in the style of the examples.

End with a Results section containing \\original{...}, \\integrand{...}, and \\variable{...}. Use \\cdot for explicit multiplication.
"""

MATH_COLD_SFT_SYSTEM_PROMPT = (
    "Return only the requested XML-style tagged response. "
    "Write the <reasoning> section as a brief planning trace of 2 to 4 short sentences. "
    "State the concrete modification, why it is harder, and the key new inference. "
    "Do not use it as a scratchpad. Do not include long derivations, alternative attempts, "
    "false starts, or self-correction chatter. Keep the <modified_solution> concise and "
    "direct, with only the steps needed to justify the final boxed answer. Prefer the "
    "shortest valid derivation."
)

INTEGRATION_SETTER_PROMPT = (
    "Q: Starting from the function {base_question} to generate a new "
    "indefinite integration problem.\n\nA: "
)

INTEGRATION_SOLVER_PROMPT = (
    "Solve the following problem step by step:\n{question}\n"
    "Ensure that each step is explained clearly. The final solution should be "
    "presented in LaTeX format, enclosed in \\boxed{}, and explicitly use "
    "\\cdot for multiplication.\n\n"
)

MATH_SOLVER_PROMPT = """Solve the following problem step by step:
{problem}
Ensure that each step is explained clearly. End with exactly one final answer in the form \\boxed{{...}}.
"""

MATH_SETTER_HARD_PROMPT = """You are an expert competition-math problem writer.

Given a seed problem and its reference solution, create a new problem-solution pair
that is distinctly harder for a strong solver while remaining valid and solvable.

Requirements:
- The new problem must be mathematically well-posed and solvable.
- The new solution must correctly solve the new problem.
- Do not make a degenerate variant whose answer is no solution, empty set, undefined, impossible, or inconsistent.
- Keep the new problem in the same broad subject and skill family as the seed.
- Require at least one extra nontrivial inference beyond the seed.
- Do not merely rename variables or swap constants.
- Prefer one clean extra inference over a broad structural generalization.
- Keep constants numeric unless the seed already uses symbolic parameters.
- Preserve the seed's core object type, scale, and construction; for geometry, do not increase the number of main objects or vertices.
- Do not introduce a new theorem, new case split, or substantially larger construction.
- If the seed is already complex, make a small same-setup variant rather than generalizing the whole problem.
- Choose a variant whose complete solution can fit in at most 8 concise sentences.
- First write a concise reasoning plan explaining the mathematical transformation.
- Keep the reasoning section under 60 words; do not include exploratory dead ends.
- Do not solve the new problem or state the new final answer in the reasoning section.
- Keep the final solution under 300 words and avoid filler outside the actual derivation.
- Do not include failed attempts, self-correction chatter, solution-side self-check prefaces, or caveats.
- The final boxed answer must fully answer every quantity requested by the problem.
- End the solution with exactly one final answer formatted as \\boxed{{...}}.

Seed Problem:
{seed_problem}

Seed Solution:
{seed_solution}

Return exactly this tagged format:
<reasoning>
Briefly explain the planned modification and why the new pair is well-posed. Keep this under 60 words.
</reasoning>
<derived_problem>
The new problem statement.
</derived_problem>
<modified_solution>
The complete solution to the new problem, under 300 words, ending with exactly one \\boxed{{...}}.
</modified_solution>
"""

MATH_SETTER_COLLECTION_HARD_PROMPT = """You are an expert competition-math problem writer.

Given a seed problem and its reference solution, create a new problem-solution pair that is harder while remaining valid and solvable.

Requirements:
- The new problem must be mathematically well-posed and solvable.
- The new solution must correctly solve the new problem.
- Keep the new problem recognizably related to the seed; do not jump to an unrelated topic or method.
- Make a real mathematical modification; do not just paraphrase the seed, rename variables.
- Do not make a degenerate variant whose intended final answer is no solution, empty set, undefined, impossible, or inconsistent.
- Write the reasoning section as 2 to 4 short sentences describing the modification plan only.
- State the concrete change, why it is harder, and the key new inference.
- Do not use the reasoning section as a scratchpad. Do not include long derivations, alternative attempts, false starts, or self-correction chatter.
- Keep the modified solution concise and direct. Avoid unnecessary exposition that is not needed to justify the final boxed answer.
- Prefer the shortest valid derivation. Avoid routine arithmetic expansion, exhaustive case enumeration, or repeating analogous steps when one compact sentence would suffice.
- The derived problem must be a clean final problem statement only.
- The modified solution must be a clean final solution only, ending with exactly one final answer formatted as \\boxed{{...}}.

Seed Problem:
{seed_problem}

Seed Solution:
{seed_solution}

Return exactly this tagged format:
<reasoning>
Two to four short sentences stating the concrete change, why it is harder, and the key new inference.
</reasoning>
<derived_problem>
The new problem statement.
</derived_problem>
<modified_solution>
The complete solution to the new problem, ending with exactly one \\boxed{{...}}.
</modified_solution>
"""

SOFT_VERIFIER_PROMPT = """You are a careful math verifier.

Verify the generated pair relative to the seed.

Use these meanings:
- valid_problem: the generated problem is mathematically well-posed, unambiguous, and has enough information to determine the requested answer.
- valid_solution: the generated solution correctly solves the generated problem, and the final boxed answer matches the solution.
- complete_final_answer: the final boxed answer fully answers every quantity requested by the generated problem.
- For this task, mark valid_problem false for degenerate variants whose intended final answer is no solution, empty set, undefined, impossible, or inconsistent.
- seed_anchored: True if the generated problem is still recognizably related to the seed and does not jump to an unrelated topic or method. Small changes in constants, framing, target quantity, nearby algebraic form, or moderate setup variation should still count as true.
- not_trivial_copy: False only for exact copies, near-copies, cosmetic paraphrases, variable renames with no meaningful mathematical change. True for any real mathematical modification, even if it is only modestly different.

Verify only validity and relation to the seed. Do not require the variant to be harder.

Seed Problem:
{seed_problem}

Seed Solution:
{seed_solution}

Derived Problem:
{derived_problem}

Modified Solution:
{modified_solution}

First think briefly step by step. Then return exactly this format:
<reasoning>
One short paragraph explaining the check.
</reasoning>
<valid_problem>true or false</valid_problem>
<valid_solution>true or false</valid_solution>
<seed_anchored>true or false</seed_anchored>
<not_trivial_copy>true or false</not_trivial_copy>
<complete_final_answer>true or false</complete_final_answer>
"""


def setter_prompt(question: str) -> str:
    return INTEGRATION_SETTER_PROMPT.replace("{base_question}", question)


def setter_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": setter_prompt(question)}]


def question_solution_prompt(question: str) -> str:
    return INTEGRATION_SOLVER_PROMPT.replace("{question}", question)


def solver_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question_solution_prompt(question)}]


def make_math_solver_prompt(problem: str) -> str:
    return MATH_SOLVER_PROMPT.format(problem=problem)


def math_solver_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": make_math_solver_prompt(problem)}]


def question_format_prompt(
    question_type: str,
    expr: str,
    var: str,
    latex: bool = False,
) -> str:
    if question_type == "integration":
        if latex:
            return f"Integrate the expression ${expr}$ with respect to ${var}$."
        return (
            f"Integrate the expression \\expression{{{expr}}} "
            f"with respect to \\variable{{{var}}}."
        )
    raise NotImplementedError(f"Question type not implemented: {question_type}")


def make_math_setter_prompt(
    seed_problem: str,
    seed_solution: str,
    *,
    objective: str = "hard",
) -> str:
    if objective != "hard":
        raise ValueError(f"Unknown objective: {objective}")
    template = MATH_SETTER_HARD_PROMPT
    return template.format(
        seed_problem=seed_problem.strip(),
        seed_solution=seed_solution.strip(),
    ).strip()


def make_math_setter_collection_prompt(
    seed_problem: str,
    seed_solution: str,
    *,
    objective: str = "hard",
) -> str:
    if objective != "hard":
        raise ValueError(f"Unknown objective: {objective}")
    template = MATH_SETTER_COLLECTION_HARD_PROMPT
    return template.format(
        seed_problem=seed_problem.strip(),
        seed_solution=seed_solution.strip(),
    )


def make_math_setter_messages(
    seed_problem: str,
    seed_solution: str,
    *,
    objective: str = "hard",
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": make_math_setter_prompt(
                seed_problem,
                seed_solution,
                objective=objective,
            ),
        }
    ]


def make_soft_verifier_prompt(
    *,
    seed_problem: str,
    seed_solution: str,
    derived_problem: str,
    modified_solution: str,
) -> str:
    return SOFT_VERIFIER_PROMPT.format(
        seed_problem=seed_problem,
        seed_solution=seed_solution,
        derived_problem=derived_problem,
        modified_solution=modified_solution,
    )
