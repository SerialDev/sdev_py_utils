from typing import List, Dict, Any, Optional
import aiohttp
import json
import re
import asyncio
from openai import AsyncOpenAI
from app.types import (
    Tool,
    MCPServerConfig,
    ToolCallResult,
    AppConfig,
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    OpenAIToolCallFunction,
)


async def get_mcp_tools_definition_list(server: MCPServerConfig) -> List[Tool]:
    """
    Fetch the list of tool definitions from a given MCP server.

    Args:
        server (MCPServerConfig): The MCP server configuration to fetch tools from.

    Returns:
        List[Tool]: A list of Tool objects defined on the server, or an empty list on error.
    """
    print(f"\033[36mFetching tools from server '{server.name}' at {server.url}\033[0m")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{server.url.rstrip('/')}/tools"
            print(f"  Constructed tools URL: {url}")
            async with session.get(url, timeout=10, server=server) as resp:
                if resp.status != 200:
                    print(
                        f"\033[31m[ERROR]\033[0m HTTP {resp.status} fetching tools from {server.name} ({url})"
                    )
                    print(f"  Response text: {await resp.text()[:500]}")
                    return []
                try:
                    json_payload = await resp.json()
                    if (
                        isinstance(json_payload, dict)
                        and json_payload.get("success")
                        and isinstance(json_payload.get("data"), dict)
                    ):
                        tools_raw = json_payload["data"].get("tools", [])
                        if not isinstance(tools_raw, list):
                            print(
                                f"\033[31m[ERROR]\033[0m 'tools' field is not a list in payload from {server.name}. Got: {type(tools_raw)}"
                            )
                            return []
                    else:
                        print(
                            f"\033[31m[ERROR]\033[0m Unexpected payload structure from {server.name}. Expected '{{success:true, data:{{tools:[]}}}}'. Got: {str(json_payload)[:500]}"
                        )
                        return []

                    tools = [Tool(**tool_data) for tool_data in tools_raw]
                    print(
                        f"Fetched {len(tools)} raw tool definitions from {server.name}"
                    )
                    return tools
                except json.JSONDecodeError as je:
                    print(
                        f"\033[31m[ERROR]\033[0m Failed to parse JSON tools from {server.name}: {je}. Response: {await resp.text()[:500]}"
                    )
                    return []
                except Exception as e:
                    print(
                        f"\033[31m[ERROR]\033[0m Error processing tool definitions from {server.name}: {type(e).__name__} - {e}"
                    )
                    return []
    except aiohttp.ClientError as e:
        print(
            f"\033[31m[AIOHTTP ERROR]\033[0m Could not connect or fetch tools from {server.name} ({server.url}): {e}"
        )
        return []
    except Exception as e:
        print(
            f"\033[31m[FRAMEWORK ERROR]\033[0m Unexpected error in get_mcp_tools_definition_list for {server.name}: {type(e).__name__} - {e}"
        )
        return []


async def execute_mcp_tool_on_server(
    server: MCPServerConfig, tool_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a tool on the specified MCP server with the given arguments.

    Args:
        server (MCPServerConfig): The MCP server configuration.
        tool_name (str): The name of the tool to execute.
        arguments (Dict[str, Any]): Arguments to pass to the tool.

    Returns:
        Dict[str, Any]: The response from the MCP server, parsed as a dictionary.
    """
    print(
        f"\033[36m  [MCP Call] Calling actual tool '{tool_name}' on server '{server.name}' with args: {arguments}\033[0m"
    )
    try:
        async with aiohttp.ClientSession() as session:
            call_url = f"{server.url.rstrip('/')}/call"
            print(f"  Constructed call URL: {call_url}")
            payload = {"tool_name": tool_name, "arguments": arguments}
            async with session.post(call_url, json=payload, timeout=120) as resp:
                response_text = await resp.text()
                try:
                    result_data = json.loads(response_text)
                    # Regardless of HTTP status, the content of result_data is what we process
                    if resp.status == 200:
                        print(
                            f"\033[32m  Tool '{tool_name}' on {server.name} HTTP 200. MCP Payload: {str(result_data)[:200]}\033[0m"
                        )
                    else:  # HTTP error
                        print(
                            f"\033[31m  [MCP Call HTTP Error] Tool '{tool_name}' on {server.name} HTTP {resp.status}. Payload: {str(result_data)[:200]}\033[0m"
                        )
                        # If HTTP error, and result_data is a dict, ensure it reflects failure if not already clear
                        if isinstance(result_data, dict):
                            if (
                                "success" not in result_data
                                or result_data.get("success") is not False
                            ):
                                result_data["success"] = (
                                    False  # Mark as failed due to HTTP
                                )
                                if "error" not in result_data:
                                    result_data["error"] = (
                                        f"MCP Call HTTP Error {resp.status}"
                                    )
                        elif not isinstance(
                            result_data, dict
                        ):  # Non-dict payload with HTTP error
                            result_data = {
                                "success": False,
                                "error": f"HTTP {resp.status} with non-dict payload",
                                "_raw_response_text": response_text,
                            }

                    if not isinstance(result_data, dict):
                        return {
                            "success": False,
                            "_raw_response_non_dict": result_data,
                            "error": f"Non-dict response loaded, HTTP {resp.status}",
                        }
                    return result_data
                except json.JSONDecodeError:
                    print(
                        f"\033[31m  [MCP Call JSON Error] Tool '{tool_name}' on {server.name}: Failed to decode JSON. HTTP {resp.status}. Response: {response_text[:500]}\033[0m"
                    )
                    return {
                        "success": False,
                        "error": f"JSON decode error from tool server (HTTP {resp.status})",
                        "_raw_response_text": response_text,
                    }
    except aiohttp.ClientError as e:
        print(
            f"\033[31m  [AIOHTTP ERROR]\033[0m Network error calling tool '{tool_name}' on {server.name}: {e}"
        )
        return {"success": False, "error": f"Network error: {str(e)}"}
    except Exception as e:
        print(
            f"\033[31m  [Framework Error]\033[0m Unexpected error in execute_mcp_tool_on_server for '{tool_name}': {type(e).__name__} {e}"
        )
        return {
            "success": False,
            "error": f"Unexpected framework error during MCP call: {str(e)}",
        }


def contains_error_keywords(text_data: str) -> bool:
    """
    Check if the given text contains common error keywords.

    Args:
        text_data (str): The text to check.

    Returns:
        bool: True if any error keywords are found, False otherwise.
    """
    if not isinstance(text_data, str):
        return False
    error_patterns = [
        r"\berror\b",
        r"\bfailed\b",
        r"\bunrecognized\b",
        r"\bexception\b",
        r"\btraceback\b",
        r"\bdenied\b",
        r"\binvalid\b",
        r"\bnot found\b",
    ]
    for pattern in error_patterns:
        if re.search(pattern, text_data, re.IGNORECASE):
            return True
    return False


async def call_tool_and_adapt_for_loop(
    server: MCPServerConfig, tool_name_on_mcp: str, arguments: Dict[str, Any]
) -> ToolCallResult:
    """
    Call a tool on the MCP server, robustly parse arguments, and adapt the response for the agentic loop.

    Args:
        server (MCPServerConfig): The MCP server configuration.
        tool_name_on_mcp (str): The name of the tool on the MCP server.
        arguments (Dict[str, Any]): Arguments to pass to the tool.

    Returns:
        ToolCallResult: A dictionary with 'mcp_succeeded' and 'llm_payload' keys, indicating success and the payload for the LLM.
    """

    # Robustly auto-parse JSON strings into dicts, universally handling all inputs:
    def robust_json_load(arg):
        if isinstance(arg, str):
            try:
                return json.loads(arg)
            except json.JSONDecodeError:
                pass  # Return original if not valid JSON
        return arg

    # Recursively handle all nested arguments
    def recursively_parse_args(arg):
        if isinstance(arg, dict):
            return {k: recursively_parse_args(v) for k, v in arg.items()}
        elif isinstance(arg, list):
            return [recursively_parse_args(elem) for elem in arg]
        else:
            return robust_json_load(arg)

    # Apply robust parsing universally to ALL arguments:
    parsed_arguments = recursively_parse_args(arguments)

    # Call MCP with guaranteed correct arguments:
    mcp_response_dict = await execute_mcp_tool_on_server(
        server, tool_name_on_mcp, parsed_arguments
    )

    effective_mcp_success = False
    llm_data_payload_for_success: Any = None
    error_message_for_llm: Optional[str] = None
    details_for_llm_on_failure: Any = mcp_response_dict

    mcp_payload_success_val = mcp_response_dict.get("success")
    if isinstance(mcp_payload_success_val, bool):
        if mcp_payload_success_val:
            effective_mcp_success = True
            llm_data_payload_for_success = mcp_response_dict.get("data") or {
                k: v for k, v in mcp_response_dict.items() if k != "success"
            }
            output_field = mcp_response_dict.get("output")
            if isinstance(output_field, str) and contains_error_keywords(output_field):
                effective_mcp_success = False
                error_message_for_llm = (
                    f"Tool output indicates error despite success flag: {output_field}"
                )
                details_for_llm_on_failure = mcp_response_dict
        else:
            effective_mcp_success = False
            error_message_for_llm = (
                mcp_response_dict.get("error")
                or mcp_response_dict.get("output")
                or f"Tool '{tool_name_on_mcp}' explicitly reported success:false."
            )
    else:
        effective_mcp_success = False
        error_message_for_llm = (
            f"Ambiguous response from tool '{tool_name_on_mcp}'. LLM to interpret."
        )

    if not effective_mcp_success:
        final_llm_payload = {
            "error": error_message_for_llm
            or f"Tool '{tool_name_on_mcp}' determined as failed by adapter.",
            "details": details_for_llm_on_failure,
        }
    else:
        final_llm_payload = llm_data_payload_for_success

    return {"mcp_succeeded": effective_mcp_success, "llm_payload": final_llm_payload}


def format_tools_for_openai_from_mcp(
    servers_mcp_tools: Dict[str, List[Tool]]
) -> List[ChatCompletionToolParam]:
    """
    Format MCP tool definitions from all servers into OpenAI-compatible tool parameters.

    Args:
        servers_mcp_tools (Dict[str, List[Tool]]): Mapping of server names to lists of Tool objects.

    Returns:
        List[ChatCompletionToolParam]: List of tool parameters formatted for OpenAI API.
    """
    all_openai_tools: List[ChatCompletionToolParam] = []
    for server_name, mcp_tool_list in servers_mcp_tools.items():
        print(
            f"  Formatting {len(mcp_tool_list)} tools from server '{server_name}' for OpenAI"
        )
        for mcp_tool_def in mcp_tool_list:
            all_openai_tools.append(mcp_tool_def.to_openai_tool_param(server_name))
    print(f"Formatted {len(all_openai_tools)} tools in total for OpenAI.")
    return all_openai_tools


async def run_agentic_loop(
    client: AsyncOpenAI,
    app_config: AppConfig,
    user_prompt: str,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """
    Main agentic loop for orchestrating LLM and tool interactions.

    Args:
        client (AsyncOpenAI): The OpenAI client for LLM calls.
        app_config (AppConfig): Application configuration, including servers and LLM settings.
        user_prompt (str): The user's initial prompt or request.
        max_steps (int, optional): Maximum number of agentic steps to run. Defaults to 10.

    Returns:
        Dict[str, Any]: Final response, history, and reason for stopping.

    Example usage:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_client = AsyncOpenAI(api_key=api_key)
        app_config = AppConfig(
            servers=[MCPServerConfig(name="local_mcp", url="http://0.0.0.0:8090/mcp", description="Local MCP Development Server for executing commands")],
            openai_model="gpt-4o",
            temperature=0.1,
            max_retries_per_tool_call=2,
        )
        result = await run_agentic_loop(client=openai_client, app_config=app_config, user_prompt=prompt, max_steps=5)
    """
    # Initialization
    servers_mcp_tools_definitions: Dict[str, List[Tool]] = {}
    if not app_config.servers:
        print(
            "\033[31m[CONFIG ERROR] No servers defined in AppConfig. Agent cannot fetch tools.\033[0m"
        )
        return {
            "final_response": "Configuration error: No MCP servers are defined.",
            "history": [],
        }

    fetch_tasks = [
        get_mcp_tools_definition_list(srv_cfg) for srv_cfg in app_config.servers
    ]
    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for i, srv_cfg in enumerate(app_config.servers):
        if isinstance(results[i], Exception):
            print(
                f"\033[31m[CRITICAL ERROR] Failed to fetch tools for server '{srv_cfg.name}': {results[i]}\033[0m"
            )
            servers_mcp_tools_definitions[srv_cfg.name] = []
        else:
            servers_mcp_tools_definitions[srv_cfg.name] = results[i]  # type: ignore

    openai_tools_list: List[ChatCompletionToolParam] = format_tools_for_openai_from_mcp(
        servers_mcp_tools_definitions
    )

    if not openai_tools_list:
        print(
            "\033[31m[CRITICAL ERROR] No tools available. Agent cannot function.\033[0m"
        )
        return {
            "final_response": "I'm sorry, no tools seem to be available.",
            "history": [],
        }

    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a highly capable assistant that uses tools to accomplish user requests. "
                "Your primary goal is to fulfill the user's request by intelligently planning and executing actions using the available tools.\n"
                "GUIDELINES FOR TOOL USE AND RETRIES:\n"
                "1. Analyze the user's request. Tools are prefixed 'serverName_toolName'.\n"
                "2. **Prioritize Action:** If a tool is needed, make a `tool_call` immediately.\n"
                "3. **Tool Execution & Results:** You will receive results as JSON in a 'tool' role message. This JSON is the `llm_payload` from the framework. \n"
                '   - Success: The `llm_payload` directly contains the tool\'s output data (e.g., `{"output": "some result"}` or `{"field": "value"}`). Use this data for next steps.\n'
                '   - Failure: The `llm_payload` will be an object like `{"error": "description", "details": {...full_mcp_response...}}`.\n'
                "4. **RETRY MECHANISM:**\n"
                "   - If a tool call fails (the framework determined it was not successful, based on explicit `success:false` from the tool OR error keywords in the output), you will receive the error details in the `llm_payload`. Then, you will get a follow-up `user` message asking you to correct and retry THAT SPECIFIC FAILED ACTION. This is an opportunity to fix arguments or try an alternative for that sub-task.\n"
                "   - You should respond with a new `tool_calls` field if you want to retry the sub-task. The `id` of this new tool call in your `tool_calls` list must be new and unique. The system will then attempt your corrected call.\n"
                "   - This retry process for a single sub-task can happen up to "
                f"{app_config.max_retries_per_tool_call} times.\n"
                "   - If you believe the sub-task is unrecoverable or don't want to retry now, respond to the retry prompt with your reasoning WITHOUT a `tool_calls` field.\n"
                "   - **After a tool's retry sequence (either success, or max retries reached, or you gave up), the main plan continues.** Analyze all outcomes and decide the next overall step.\n"
                "5. **Continue on Partial Failure:** Do not stop the entire plan on one tool's final failure if other parts can proceed or alternatives exist for the overall goal.\n"
                "6. **Final Summary:** When the whole request is done or no more progress can be made, provide a final concise summary. Do NOT request tool calls in the final summary."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    history = []

    for step_count in range(max_steps):
        print(f"\n--- Step {step_count+1}/{max_steps} ---")
        current_step_record = {
            "step": step_count + 1,
            "llm_interactions": [],
            "tool_processing_summary_for_step": [],
        }

        print(f"  [LLM Call] Processing with {len(messages)} messages in context...")
        # --- BEGIN ADDED LOGGING ---
        print(
            f"\033[94m    Current LLM context before main step call (Step {step_count+1}):\n{json.dumps(messages, indent=2, default=str)}\033[0m"
        )
        # --- END ADDED LOGGING ---
        llm_interaction_log: Dict[str, Any] = {
            "request_messages_context_count": len(messages)
        }

        try:
            response = await client.chat.completions.create(
                model=app_config.openai_model,
                temperature=app_config.temperature,
                messages=messages,
                tools=openai_tools_list,
                tool_choice="auto",  # type: ignore
            )
        except Exception as e:
            print(
                f"\033[31m  [LLM Call Error] OpenAI API call failed: {type(e).__name__} - {e}\033[0m"
            )
            llm_interaction_log["error"] = f"OpenAI API call failed: {e}"
            current_step_record["llm_interactions"].append(llm_interaction_log)
            history.append(current_step_record)
            return {
                "final_response": f"Error communicating with AI model: {e}",
                "history": history,
            }

        llm_response_msg = response.choices[0].message
        llm_interaction_log["response_raw_model_dump"] = llm_response_msg.model_dump(
            exclude_unset=True
        )

        assistant_response_content = llm_response_msg.content or ""
        llm_interaction_log["assistant_response_content"] = assistant_response_content

        assistant_msg_for_history: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": assistant_response_content,
        }

        initial_tool_calls_requested: List[ChatCompletionMessageToolCall] = []
        if llm_response_msg.tool_calls:
            initial_tool_calls_requested = [
                ChatCompletionMessageToolCall(
                    id=tc.id,
                    function=OpenAIToolCallFunction(
                        name=tc.function.name, arguments=tc.function.arguments
                    ),
                    type="function",
                )
                for tc in llm_response_msg.tool_calls
                if tc.function
            ]
            if initial_tool_calls_requested:
                assistant_msg_for_history["tool_calls"] = initial_tool_calls_requested

        messages.append(assistant_msg_for_history)
        llm_interaction_log["assistant_tool_calls_requested_initial"] = [
            {"id": tc.id, "name": tc.function.name, "args": tc.function.arguments}
            for tc in initial_tool_calls_requested
        ]
        current_step_record["llm_interactions"].append(llm_interaction_log)

        print(
            f"  [LLM Response] Content: {assistant_response_content if assistant_response_content else 'None'}"
        )
        if initial_tool_calls_requested:
            print(
                f"  [LLM Response] Initial Tool Calls Requested: {json.dumps(llm_interaction_log['assistant_tool_calls_requested_initial'], indent=2, default=str)}"
            )
        else:
            print("  [LLM Response] No tool calls requested by LLM.")
            history.append(current_step_record)
            print(
                f"\n[Final Agent Response at Step {step_count+1}] {assistant_response_content or 'No textual content.'}"
            )
            return {
                "final_response": assistant_response_content,
                "history": history,
                "reason_for_stop": "LLM provided response without tool calls.",
            }

        all_outcomes_for_this_step = []
        for original_tool_call_request in initial_tool_calls_requested:
            current_tool_call_id_for_attempt = original_tool_call_request.id
            current_tool_function_name = original_tool_call_request.function.name
            current_tool_function_args_str = (
                original_tool_call_request.function.arguments
            )

            original_request_processing_log = {
                "original_tool_call_id": original_tool_call_request.id,
                "original_function_name": current_tool_function_name,
                "attempts": [],
            }
            tool_succeeded_for_original_goal = False

            for attempt_num in range(
                1, app_config.max_retries_per_tool_call + 2
            ):  # +1 for initial, +1 for range end
                is_retry_attempt = attempt_num > 1
                attempt_log: Dict[str, Any] = {
                    "attempt_number": attempt_num,
                    "tool_call_id_used_for_attempt": current_tool_call_id_for_attempt,
                    "function_name_attempted": current_tool_function_name,
                    "function_args_attempted": current_tool_function_args_str,
                }
                print(
                    f"\n  --- Tool Execution {'Retry ' if is_retry_attempt else ''}Attempt {attempt_num} (max {app_config.max_retries_per_tool_call+1}) for original goal of '{original_tool_call_request.function.name}' (ID '{original_tool_call_request.id}') ---"
                )
                print(
                    f"    Using Tool Call ID: {current_tool_call_id_for_attempt}, Function: {current_tool_function_name}, Args: {current_tool_function_args_str}"
                )

                tool_output_llm_payload: Dict[str, Any]
                mcp_actually_succeeded: bool = False

                try:
                    target_server_cfg: Optional[MCPServerConfig] = None
                    actual_mcp_tool_name: Optional[str] = None
                    for s_cfg in app_config.servers:
                        if current_tool_function_name.startswith(s_cfg.name + "_"):
                            target_server_cfg = s_cfg
                            actual_mcp_tool_name = current_tool_function_name[
                                len(s_cfg.name) + 1 :
                            ]
                            break
                    if not target_server_cfg or not actual_mcp_tool_name:
                        raise ValueError(
                            f"Could not route OpenAI tool '{current_tool_function_name}'."
                        )

                    args_dict = json.loads(current_tool_function_args_str)
                    tool_call_outcome: ToolCallResult = (
                        await call_tool_and_adapt_for_loop(
                            target_server_cfg, actual_mcp_tool_name, args_dict
                        )
                    )
                    mcp_actually_succeeded = tool_call_outcome["mcp_succeeded"]
                    tool_output_llm_payload = tool_call_outcome["llm_payload"]
                    attempt_log["adapted_mcp_result"] = tool_call_outcome

                except Exception as e_tool_exec:
                    print(
                        f"\033[31m    [Framework Error] Processing tool {current_tool_function_name}: {e_tool_exec}\033[0m"
                    )
                    mcp_actually_succeeded = False
                    tool_output_llm_payload = {
                        "error": f"Framework error processing tool call: {str(e_tool_exec)}",
                        "details": {"exception_type": type(e_tool_exec).__name__},
                    }
                    attempt_log["framework_error"] = str(e_tool_exec)

                tool_output_content_str = json.dumps(
                    tool_output_llm_payload, default=str
                )

                print(
                    f"    [Tool Call Result (Attempt {attempt_num})] ID: {current_tool_call_id_for_attempt}"
                )
                print(
                    f"      MCP Succeeded Flag (Adapter determined): {mcp_actually_succeeded}"
                )
                print(
                    f"      Payload for LLM: {json.dumps(tool_output_llm_payload, indent=2, default=str)}"
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": current_tool_call_id_for_attempt,
                        "content": tool_output_content_str,
                    }
                )
                attempt_log["tool_message_added_to_context"] = messages[-1]
                original_request_processing_log["attempts"].append(attempt_log)

                if mcp_actually_succeeded:
                    print(
                        f"\033[32m    Tool call (ID {current_tool_call_id_for_attempt}) for original goal '{original_tool_call_request.function.name}' SUCCEEDED on attempt {attempt_num}.\033[0m"
                    )
                    tool_succeeded_for_original_goal = True
                    break

                if (
                    attempt_num <= app_config.max_retries_per_tool_call
                ):  # Check if more retries are allowed for THIS original tool call
                    print(
                        f"\033[33m    Tool call (ID {current_tool_call_id_for_attempt}) FAILED. Asking LLM for correction (Attempt {attempt_num} of {app_config.max_retries_per_tool_call + 1} total attempts for original goal).\033[0m"
                    )

                    retry_user_prompt = (
                        f"The previous tool call (ID: '{current_tool_call_id_for_attempt}', Function: '{current_tool_function_name}') failed. "
                        f"The result was: {json.dumps(tool_output_llm_payload, default=str, indent=2)}. "
                        f"This was attempt {attempt_num} to achieve the goal of the original tool request (Original ID '{original_tool_call_request.id}', Function '{original_tool_call_request.function.name}'). "
                        f"You have {app_config.max_retries_per_tool_call - attempt_num + 1} more attempts left for this specific goal. "  # Corrected remaining attempts calculation
                        "Please analyze the error and the original arguments. "
                        "To retry, provide a `tool_calls` field with ONE new or corrected tool call (ensure it has a new unique `id`). "
                        "If you believe this sub-task is unrecoverable or don't want to retry now, respond with your reasoning WITHOUT a `tool_calls` field."
                    )
                    messages.append({"role": "user", "content": retry_user_prompt})
                    attempt_log["retry_prompt_to_llm"] = messages[-1]

                    print(
                        f"      [LLM Call for Retry Guidance] Context size: {len(messages)}"
                    )
                    # --- BEGIN ADDED LOGGING ---
                    print(
                        f"\033[94m        Current LLM context before retry guidance call (Original ID '{original_tool_call_request.id}', Attempt {attempt_num}):\n{json.dumps(messages, indent=2, default=str)}\033[0m"
                    )
                    # --- END ADDED LOGGING ---
                    retry_llm_interaction_log = {
                        "request_messages_context_count": len(messages)
                    }
                    try:
                        retry_guidance_response = await client.chat.completions.create(
                            model=app_config.openai_model,
                            temperature=app_config.temperature,
                            messages=messages,
                            tools=openai_tools_list,
                            tool_choice="auto",  # type: ignore
                        )
                        llm_retry_guidance_msg = retry_guidance_response.choices[
                            0
                        ].message
                        retry_llm_interaction_log["response_raw_model_dump"] = (
                            llm_retry_guidance_msg.model_dump(exclude_unset=True)
                        )

                        guidance_content = llm_retry_guidance_msg.content or ""
                        guidance_msg_for_history: ChatCompletionMessageParam = {
                            "role": "assistant",
                            "content": guidance_content,
                        }

                        suggested_retry_calls: List[ChatCompletionMessageToolCall] = []
                        if llm_retry_guidance_msg.tool_calls:
                            suggested_retry_calls = [
                                ChatCompletionMessageToolCall(
                                    id=tc.id,
                                    function=OpenAIToolCallFunction(
                                        name=tc.function.name,
                                        arguments=tc.function.arguments,
                                    ),
                                    type="function",
                                )
                                for tc in llm_retry_guidance_msg.tool_calls
                                if tc.function
                            ]
                            if suggested_retry_calls:
                                guidance_msg_for_history["tool_calls"] = (
                                    suggested_retry_calls
                                )

                        messages.append(guidance_msg_for_history)
                        retry_llm_interaction_log[
                            "llm_retry_guidance_added_to_context"
                        ] = messages[-1]
                        print(
                            f"      [LLM Retry Guidance] Content: {guidance_content if guidance_content else 'None'}"
                        )

                        if suggested_retry_calls and len(suggested_retry_calls) == 1:
                            new_call = suggested_retry_calls[0]
                            print(
                                f"      LLM suggests retrying with new call ID: {new_call.id}, Func: {new_call.function.name}"
                            )
                            # Update current_tool_... vars for the NEXT iteration of this inner loop (the retry attempt)
                            current_tool_call_id_for_attempt = new_call.id
                            current_tool_function_name = new_call.function.name
                            current_tool_function_args_str = new_call.function.arguments
                        else:
                            print(
                                f"\033[33m      LLM did not provide a single tool call for retry. Aborting retries for original goal '{original_tool_call_request.function.name}'.\033[0m"
                            )
                            if suggested_retry_calls:
                                print(
                                    f"      LLM suggested {len(suggested_retry_calls)} calls, expected 1 for direct retry."
                                )
                            tool_succeeded_for_original_goal = (
                                False  # Mark as failed for this original goal
                            )
                            break  # Break from the retry attempts loop for this original_tool_call_request
                    except Exception as e_llm_retry:
                        print(
                            f"\033[31m      [LLM Call Error] for retry guidance: {e_llm_retry}. Aborting retries for original goal '{original_tool_call_request.function.name}'.\033[0m"
                        )
                        retry_llm_interaction_log["error"] = str(e_llm_retry)
                        tool_succeeded_for_original_goal = False  # Mark as failed
                        break  # Break from the retry attempts loop
                    finally:
                        attempt_log["llm_interaction_for_retry_guidance"] = (
                            retry_llm_interaction_log
                        )
                else:  # Max retries for this specific original tool call have been used up
                    print(
                        f"\033[31m    Max attempts ({app_config.max_retries_per_tool_call + 1}) reached for original goal '{original_tool_call_request.function.name}'. Last status was failure.\033[0m"
                    )
                    tool_succeeded_for_original_goal = (
                        False  # Ensure it's marked as failed
                    )
                    break  # Break from the retry attempts loop

            original_request_processing_log["final_success_status"] = (
                tool_succeeded_for_original_goal
            )
            all_outcomes_for_this_step.append(original_request_processing_log)

        current_step_record["tool_processing_summary_for_step"] = (
            all_outcomes_for_this_step
        )
        history.append(current_step_record)

        if step_count == max_steps - 1:
            print(
                f"\033[33m  [Warning] Reached max_steps ({max_steps}). Further LLM processing of last actions will not occur in this run.\033[0m"
            )

    print(
        f"\n[MAX STEPS REACHED or Agent Stopped] Loop concluded after {len(history)} actual steps executed."
    )
    final_response_text = "Agent stopped. Review history for details."
    # Check the very last message in the history. If it's an assistant message without tool calls, it's likely the final summary.
    if (
        messages
        and messages[-1]["role"] == "assistant"
        and not messages[-1].get("tool_calls")
    ):
        final_response_text = (
            messages[-1]["content"] or "Assistant provided no final textual content."
        )

    return {
        "final_response": final_response_text,
        "history": history,
        "reason_for_stop": f"Max steps ({max_steps}) reached or agent decided to stop.",
    }
