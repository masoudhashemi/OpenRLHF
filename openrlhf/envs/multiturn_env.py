import torch
from typing import Any, Dict, Tuple
import re
import io
import sys
import ast # Added for function calling

# --- Global Configuration ---
ENVIRONMENT_MODE = "math"  # Can be "math" or "function_calling"
MAX_TURNS = 5

# --- Math Environment Specifics ---
MATH_PROBLEM_STATEMENT = "What is 2 + 2?"
MATH_CORRECT_ANSWER = 4

# --- Function Calling Environment Specifics ---
FUNCTION_CALLING_PROBLEM_STATEMENT = (
    "You have access to a calculator and a search engine. "
    "Your task is to find out what 2*7 is and who the current UK prime minister is. "
    "Respond with [TOOL_CALL] tool_name(args_str) to use tools or [FINAL_ANSWER] your_answer to give the final answer."
)

# --- Shared State ---
current_turn = 0


# --- Mock Tools for Function Calling ---
def calculator(expression_str: str):
    try:
        # WARNING: eval is dangerous with untrusted input.
        # For a real scenario, use a safer expression parser.
        # For this controlled environment, we assume simple expressions.
        return eval(expression_str)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def search(query_str: str):
    # Mock search results
    if "weather in london" in query_str.lower():
        return "The weather in London is sunny."
    elif "uk prime minister" in query_str.lower():
        return "The current UK prime minister is Rishi Sunak."
    else:
        return "Sorry, I couldn't find information for that query."

mock_tools = {
    "calculator": calculator,
    "search": search
}


async def step(state: str, action: str, **kwargs) -> Tuple[torch.Tensor, str, bool, Dict[str, Any]]:
    """Executes one step of the environment based on ENVIRONMENT_MODE.

    Args:
        state: The current state of the environment (ongoing conversation).
        action: The action taken by the agent (LLM response).

    Returns:
        Tuple[torch.Tensor, str, bool, Dict[str, Any]]: (reward, next_state, done, info)
    """
    global current_turn, MAX_TURNS, ENVIRONMENT_MODE
    global MATH_PROBLEM_STATEMENT, MATH_CORRECT_ANSWER
    global FUNCTION_CALLING_PROBLEM_STATEMENT

    current_turn += 1
    done = False
    reward = 0.0  # Default reward
    extra_info: Dict[str, Any] = {}

    if ENVIRONMENT_MODE == "math":
        print("[multiturn_env.py] Math environment step function called.") # Added for verification
        code_match = re.search(r"```python\n(.*?)\n```", action, re.DOTALL)

        if not code_match:
            reward = -0.2
            next_state = state + action + "\nNo Python code found. Please provide code in ```python ... ``` format.\n"
        else:
            extracted_code = code_match.group(1)
            exec_globals: Dict[str, Any] = {}

            def submit_answer(ans):
                exec_globals['final_answer'] = ans

            exec_globals['submit_answer'] = submit_answer

            code_output = io.StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = code_output
            sys.stderr = code_output
            
            execution_succeeded = False
            try:
                exec(extracted_code, exec_globals)
                execution_result = code_output.getvalue()
                execution_succeeded = True 
            except Exception as e:
                execution_result = str(e)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            next_state = state + action + "\nExecution Result:\n" + execution_result + "\n"

            final_answer = exec_globals.get('final_answer')
            if final_answer == MATH_CORRECT_ANSWER:
                reward = 1.0
                done = True
            elif execution_succeeded:
                reward = 0.1
            else: 
                reward = -0.5
        
    elif ENVIRONMENT_MODE == "function_calling":
        final_answer_match = re.search(r"\[FINAL_ANSWER\] (.*)", action, re.DOTALL)
        tool_call_match = re.search(r"\[TOOL_CALL\] (\w+)\((.*)\)", action, re.DOTALL)

        if final_answer_match:
            final_answer_content = final_answer_match.group(1).strip()
            # Task: "What is 2*7 and who is the UK prime minister?"
            # Correct: "14" and "Rishi Sunak"
            if "14" in final_answer_content and "Rishi Sunak" in final_answer_content:
                reward = 1.0
            else:
                reward = -0.1
            done = True
            next_state = state + action + "\nOutcome: Final answer processed.\n"
        
        elif tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            tool_args_str = tool_call_match.group(2).strip()

            if tool_name in mock_tools:
                try:
                    # Simplistic argument parsing: assuming single string argument.
                    # For more complex args, ast.literal_eval might be used carefully.
                    if tool_args_str.startswith("'") and tool_args_str.endswith("'"):
                        parsed_args = [tool_args_str[1:-1]]
                    elif tool_args_str.startswith('"') and tool_args_str.endswith('"'):
                        parsed_args = [tool_args_str[1:-1]]
                    else: # Treat as unquoted string, or number for calculator
                        if tool_name == "calculator":
                             parsed_args = [tool_args_str] # eval will handle it
                        else:
                             parsed_args = [tool_args_str]


                    tool_result = mock_tools[tool_name](*parsed_args)
                    execution_output = str(tool_result)
                    reward = 0.2  # Successful tool call
                except Exception as e:
                    execution_output = f"Error executing tool {tool_name} with args '{tool_args_str}': {str(e)}"
                    reward = -0.3 # Tool execution error
                next_state = state + action + f"\nTool Output: {execution_output}\n"
            else:
                execution_output = f"Error: Tool '{tool_name}' not found."
                reward = -0.5 # Invalid tool name
                next_state = state + action + f"\nTool Output: {execution_output}\n"
        else:
            next_state = state + action + "\nNo valid tool call or final answer format found. Use [TOOL_CALL] tool_name(args) or [FINAL_ANSWER] your answer.\n"
            reward = -0.2 # Malformed action

    else: # Should not happen if ENVIRONMENT_MODE is set correctly
        next_state = state + action + "\nERROR: Invalid ENVIRONMENT_MODE set.\n"
        reward = -5.0 # Severe error
        done = True


    # Check MAX_TURNS for non-math modes or math modes that didn't end
    if not done and current_turn >= MAX_TURNS:
        done = True
        # Apply penalty for running out of turns if no specific reward was set for this turn's outcome
        # or if the reward was for a non-terminal event.
        if ENVIRONMENT_MODE == "math" and (reward == 0.0 or reward == 0.1):
             reward = -1.0
        elif ENVIRONMENT_MODE == "function_calling" and reward <= 0.2: # Any non-final positive or any negative reward
             reward = -1.0


    if done:
        current_turn = 0  # Reset for next episode

    return torch.tensor(reward, dtype=torch.float), next_state, done, extra_info
