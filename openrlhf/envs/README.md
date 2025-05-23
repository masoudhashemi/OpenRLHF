# Multi-Turn Environment for OpenRLHF (`multiturn_env.py`)

## Purpose

This environment (`multiturn_env.py`) is designed to facilitate the training of Large Language Models (LLMs) in an interactive, multi-turn setting using the OpenRLHF framework. It allows an LLM to engage in a dialogue or a sequence of actions to solve a task, receiving feedback (rewards) and updated states from the environment at each turn. This is particularly useful for tasks like:

*   Solving math problems step-by-step, where the LLM generates code that is executed by the environment.
*   Using external tools via function calls, where the LLM decides which tool to call and with what arguments.

The core idea is to train the LLM through reinforcement learning, where its actions (text generations) are evaluated by the environment, and the resulting experiences are used to improve its policy.

## How to Use

The `multiturn_env.py` script currently supports two primary modes, controlled by the global variable `ENVIRONMENT_MODE` within the script.

### 1. Setting the Environment Mode

Modify the `ENVIRONMENT_MODE` variable at the top of `openrlhf/envs/multiturn_env.py`:

*   `ENVIRONMENT_MODE = "math"`: For solving math problems.
*   `ENVIRONMENT_MODE = "function_calling"`: For tasks requiring tool/function calls.

### 2. Interaction Details

#### a. Math Mode (`ENVIRONMENT_MODE = "math"`)

*   **Initial State**: The environment starts with a problem statement (e.g., "What is 2 + 2?"). This is defined by the `PROBLEM_STATEMENT` global variable in the script.
*   **LLM Action Format**: The LLM is expected to generate Python code blocks to solve the problem. The environment extracts code enclosed in triple backticks:
    ```
    Some thought process...
    ```python
    # python code
    print(2 + 2)
    submit_answer(4)
    ```
    ```
*   **Code Execution**: The extracted Python code is executed. A special function `submit_answer(answer)` is available within the execution scope. The LLM should call this function to provide its final answer.
*   **State Update**: The environment responds with the execution result (stdout/stderr of the code, or an error message). This, along with the history, forms the new state.
*   **Rewards**:
    *   Correct final answer (via `submit_answer`): +1.0 (ends episode)
    *   Code executes successfully but isn't a correct final answer: +0.1
    *   No Python code found in LLM action: -0.2
    *   Code execution error: -0.5
    *   Max turns reached without correct answer: -1.0 (ends episode)

#### b. Function Calling Mode (`ENVIRONMENT_MODE = "function_calling"`)

*   **Initial State**: The environment starts with a task description that requires tool usage (e.g., "What is 2*7 and who is the UK prime minister?"). This is defined by the `FUNCTION_CALLING_PROBLEM_STATEMENT` global variable.
*   **LLM Action Format**:
    *   To use a tool: `[TOOL_CALL] tool_name(arguments_string)`
        *   Example: `[TOOL_CALL] calculator("2*7")`
        *   Example: `[TOOL_CALL] search("current UK prime minister")`
    *   To provide a final answer: `[FINAL_ANSWER] Your final textual answer.`
        *   Example: `[FINAL_ANSWER] 2*7 is 14 and the UK prime minister is Rishi Sunak.`
*   **Mock Tools**: The environment provides mock tools:
    *   `calculator(expression_str)`: Evaluates a Python numerical expression.
    *   `search(query_str)`: Returns predefined search results for specific queries.
*   **State Update**: The environment responds with the tool's output or an error message. This, along with the history, forms the new state.
*   **Rewards**:
    *   Correct final answer (matching expected keywords for the example problem): +1.0 (ends episode)
    *   Incorrect final answer: -0.1 (ends episode)
    *   Successful tool call: +0.2
    *   Malformed tool call or final answer format: -0.2
    *   Tool execution error (e.g., bad expression for calculator): -0.3
    *   Unknown tool name: -0.5
    *   Max turns reached: -1.0 (ends episode)

### 3. Integration with Training Script

To use this environment for training, you'll need to configure the PPO training script:

*   **Training Script**: `examples/scripts/train_ppo_multiturn_env.sh` is provided as an example.
*   **Agent Path**: Specify the path to the environment script using the argument:
    `--agent_func_path openrlhf/envs/multiturn_env.py`
*   **Asynchronous Training (Crucial!)**: This environment interaction model **requires** the asynchronous PPO pipeline. Ensure the training script includes the flag:
    `--async_train`
*   **Prompt Datasets**:
    *   For math mode, use: `--prompt_data openrlhf/datasets/multiturn_math_prompts.jsonl`
        *   This file should contain initial math problems, e.g., `{"input": "What is 2 + 2?"}`
    *   For function calling mode, use: `--prompt_data openrlhf/datasets/multiturn_function_calling_prompts.jsonl`
        *   This file should contain initial task descriptions, e.g., `{"input": "Your task is to find out what 2*7 is..."}`
    *   The `--input_key` argument for `train_ppo_ray.py` should match the key in your JSONL file (e.g., `"input"`).

## Interaction Mechanism & Loss Masking

The multi-turn interaction with this environment is handled by OpenRLHF's asynchronous PPO components:

1.  `LLMRayActorAsync`: This vLLM engine actor loads the `step` function from `agent_func_path`. It iteratively:
    a. Prompts the LLM with the current state.
    b. Receives the LLM's generated `action`.
    c. Records the character offsets of this `action` within the accumulating dialogue history (`action_ranges`).
    d. Calls the environment's `step(state, action)` function.
    e. Receives `reward`, `next_state`, and `done` from the environment.
    f. Updates `state` to `next_state` and repeats if not `done`.

2.  `SamplesGeneratorAsync`: This component receives the final, complete dialogue history (`state`) and the `action_ranges` from `LLMRayActorAsync`.
    *   It tokenizes the entire dialogue history.
    *   Using the `action_ranges` (converted to token spans), it creates a token-level `action_mask`. This mask has `1`s for tokens generated by the LLM and `0`s for tokens from the environment's responses or the initial prompt.

3.  **Loss Calculation**: The PPO algorithm then uses this `action_mask` during the policy loss calculation. This ensures that gradients are only computed for the LLM's own generated tokens, effectively "masking out" the environment's responses from the loss. This trains the LLM to choose better actions, not to mimic the environment.

This setup allows for robust training of LLMs in complex, interactive scenarios.
