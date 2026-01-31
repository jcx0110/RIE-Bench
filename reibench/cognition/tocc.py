"""
TOCC (Task-Oriented Cognitive Control) processing logic.
Handles referring expression resolution for vague instructions.
"""

from guidance import gen


def process_tocc_reference(planner_model, query, prev_steps):
    """
    Process TOCC referring expression resolution.
    
    Args:
        planner_model: The LLM model for generating clarifications
        query: The original human instruction
        prev_steps: Previous steps in the plan
        
    Returns:
        tuple: (clarified_instruction, prompt_reference)
    """
    if len(prev_steps) == 0:
        TOCC_hint = f"""
            Human pending instruction may contain vague referring expressions, such as ``electronic devices'', ``beverages'', ``fruits'', and ``containers'', which are not specific items. \n
            You are a robot, your task is to make the `Human Pending Instruction" clear. \n
            Do not add extra commentary or conversation or the hole plan, only output the clear instruction. \n
            Use the previous context below to resolve the referring expressions:\n
            Previous context:\n
            {query.strip()}\n
            Please make the `Human Pending Instruction" clear:"""
        prompt_reference = planner_model + f"{TOCC_hint}\n" + gen(stop='.')
        prompt_references = str(prompt_reference).split('\n')
        prompt_reference = prompt_references[-1].strip()
        with open("output.txt", "a") as f:
            f.write(f"{str(prompt_reference)}\n")
        return prompt_reference, prompt_reference
    else:
        return None, None

