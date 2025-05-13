""

from logging import logger
from typing import Optional, Any, Dict, List
import io
import contextlib
import traceback
import textwrap
import time
from typing import TypedDict
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    max_tokens=100,
    messages=[
        {
            "role": "system",
            "content": """You are a helpful assistant that takes inspiration from world class developers like
         John Carmack, Casey Muratori and, Jonathan Blow, you will give me code recommendations heavily influenced
         by data oriented design maxims, and high quality performant code """,
        },
        {"role": "user", "content": "Can you show me a good example of code"},
    ],
)

dir(response)
response.values()


def test():
    pass


# ------------------------------------------------------------------------- #


class REPLHistory(TypedDict):
    timestamp: List[float]
    command: List[str]
    output: List[str]
    error: List[str]
    success: List[bool]


class PythonREPL:
    env: Dict[str, Any]
    history: REPLHistory
    macros: Dict[str, str]
    watchlist: List[str]

    def __init__(self):
        self.env = {}
        self.history: REPLHistory = {
            "timestamp": [],
            "command": [],
            "output": [],
            "error": [],
            "success": [],
        }

        self.macros = {}
        self.watchlist = []

    def _snapshot(self) -> dict[str, Any]:
        return self.env.copy()

    def get_variable(self, name: str) -> Optional[object]:
        import decimal

        val = self.env.get(name)
        return float(val) if isinstance(val, decimal.Decimal) else val

    def _restore(self, snap: dict[str, Any]) -> None:
        self.env.clear()
        self.env.update(snap)

    def execute(self, code: str) -> Dict[str, Optional[str]]:
        code_clean: str = textwrap.dedent(code).strip()
        logger.debug("\033[36m[EXECUTE]\033[0m", repr(code_clean))
        out_buf: io.StringIO = io.StringIO()
        err_buf: io.StringIO = io.StringIO()
        snap: Dict[str, Any] = self._snapshot()
        succeeded: bool = True
        is_expr: bool = False

        try:
            compiled = compile(code_clean, "<repl>", "eval")
            is_expr = True
        except SyntaxError:
            try:
                compiled = compile(code_clean, "<repl>", "exec")
                is_expr = False
            except SyntaxError as e:
                logger.debug(
                    "\033[31m[SYNTAX ERROR]\033[0m", f"{e.msg} at line {e.lineno}"
                )
                self.history["timestamp"].append(time.time())
                self.history["command"].append(code_clean)
                self.history["output"].append("")
                self.history["error"].append(f"SyntaxError: {e.msg} at line {e.lineno}")
                self.history["success"].append(False)
                return {
                    "output": "",
                    "error": f"SyntaxError: {e.msg} at line {e.lineno}",
                }

        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            try:
                if is_expr:
                    result = eval(compiled, self.env, self.env)
                    if result is not None:
                        logger.debug(result)
                else:
                    exec(compiled, self.env, self.env)
            except Exception:
                self._restore(snap)
                succeeded = False
                traceback.print_exc()

        if succeeded:
            logger.debug("\033[32m[COMMIT]\033[0m")
        else:
            logger.debug("\033[31m[ROLLBACK]\033[0m")

        out: str = out_buf.getvalue().rstrip()
        err: Optional[str] = err_buf.getvalue().rstrip() or None

        self.history["timestamp"].append(float(time.time()))
        self.history["command"].append(str(code_clean or ""))
        self.history["output"].append(str(out or ""))
        self.history["error"].append(str(err or ""))
        self.history["success"].append(bool(succeeded))

        return {"output": out, "error": err}

    def variables(self) -> dict[str, Any]:
        import decimal

        def json_safe(val: Any) -> Any:
            if isinstance(val, decimal.Decimal):
                return float(val)
            return val

        return {
            k: json_safe(v)
            for k, v in self.env.items()
            if not k.startswith("__") and not callable(v)
        }

    def who(self) -> None:
        for k, v in self.variables().items():
            logger.debug(f"\033[35m{str(k):<20}\033[0m {type(v).__name__}")

    def define_macro(self, name: str, code: str) -> None:
        self.macros[name] = code.strip()

    def invoke_macro(self, name: str) -> dict[str, Optional[str]]:
        return self.execute(self.macros.get(name, ""))

    def globals_view(self) -> dict[str, Any]:
        return dict(self.env)

    def tool_info(self) -> dict:
        return {
            "execute": {
                "description": """
Execute a string of Python code. Supports both expressions and blocks.
""",
                "args": ["code: str"],
                "returns": "dict with keys 'output', 'error'",
                "tags": ["core", "mutation"],
            },
            "variables": {
                "description": """
Get current global variable bindings (non-callables only).
""",
                "args": [],
                "returns": "Dict[str, Any]",
                "tags": ["read"],
            },
            "who": {
                "description": """
Print all current non-callable variables and their types.
""",
                "args": [],
                "returns": "None",
                "tags": ["read", "UI"],
            },
            "define_macro": {
                "description": "Define a named macro from a code block.",
                "args": ["name: str", "code: str"],
                "returns": "None",
                "tags": ["macro", "mutation"],
            },
            "invoke_macro": {
                "description": "Run a previously defined macro by name.",
                "args": ["name: str"],
                "returns": "Execution result dict",
                "tags": ["macro"],
            },
            "globals_view": {
                "description": "Return all current globals.",
                "args": [],
                "returns": "Dict[str, Any]",
                "tags": ["read"],
            },
        }


def to_openai_tools(repl: PythonREPL) -> List[Dict[str, Any]]:
    tools = []
    for name, meta in repl.tool_info().items():
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        for arg in meta["args"]:
            if ":" in arg:
                arg_name, arg_type = [s.strip() for s in arg.split(":")]
                json_type = (
                    "string"
                    if "str" in arg_type
                    else "boolean" if "bool" in arg_type else "number"
                )
                schema["function"]["parameters"]["properties"][arg_name] = {
                    "type": json_type
                }
                schema["function"]["parameters"]["required"].append(arg_name)
            else:
                schema["function"]["parameters"]["properties"][arg] = {"type": "string"}
                schema["function"]["parameters"]["required"].append(arg)
        tools.append(schema)
    return tools


def run_openai_repl_round(
    client: Any,
    repl: PythonREPL,
    user_prompt: str,
    model: str = "gpt-4-1106-preview",
    max_steps: int = 5,
) -> Dict[str, Any]:
    import json
    from io import StringIO
    import contextlib

    history = []

    def log(msg, color="0"):
        print(f"\033[{color}m{msg}\033[0m")

    def get_tools_info(repl):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            arg.split(":")[0]: {"type": "string"}
                            for arg in info["args"]
                        },
                        "required": [arg.split(":")[0] for arg in info["args"]],
                    },
                },
            }
            for name, info in repl.tool_info().items()
        ]

    def normalize_content(content):
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return content

    tools = get_tools_info(repl)

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        repl.who()
    who_state = buf.getvalue().strip()

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                """
You're a Python assistant with access to a persistent in-memory Python REPL.
"""
                "You can use tools to run code, inspect variables, or call macros.\n\n"
                f"Current known variables:\n{who_state or '(none yet)'}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    for step in range(max_steps):
        log(f"[STEP {step+1} CONTEXT]\n{json.dumps(messages, indent=2)}", color="34")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        content_text = normalize_content(message.content)

        step_record = {
            "step": step + 1,
            "context": messages.copy(),
            "response": content_text,
        }

        if not message.tool_calls:
            log(f"[FINAL RESPONSE] {content_text}", color="32")
            history.append(step_record)
            return {"final_response": content_text, "history": history}

        ephemeral_errors = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except Exception as e:
                log(f"[ERROR] Failed to decode tool args: {e}", color="31")
                ephemeral_errors.append(f"Invalid tool arguments: {e}")
                continue

            log(f"[AGENT CALL] {tool_name}({args})", color="36")

            try:
                raw_result = getattr(repl, tool_name)(**args)
                result = (
                    raw_result
                    if isinstance(raw_result, dict)
                    else {"result": json.dumps(raw_result, default=str)}
                )

            except Exception as e:
                result = {"error": str(e)}

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                }
            )
            content_str = json.dumps(result, default=str)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": content_str,
                }
            )

            if isinstance(result, dict) and result.get("error"):
                error_msg = result["error"]
                log(f"[ERROR] {error_msg}", color="31")
                ephemeral_errors.append(
                    f"""
The previous command resulted in an error: {error_msg}. Please
resolve this issue.
"""
                )

        if ephemeral_errors:
            messages.append({"role": "user", "content": " ".join(ephemeral_errors)})
        else:
            messages = [
                msg
                for msg in messages
                if not (
                    msg["role"] == "user"
                    and isinstance(msg["content"], str)
                    and "resulted in an error" in msg["content"]
                )
            ]

        history.append(step_record)

    log("[ERROR] Reached max steps without concluding.", color="31")
    return {"final_response": None, "history": history}
