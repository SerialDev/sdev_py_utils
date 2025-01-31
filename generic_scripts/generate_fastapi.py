import os
import argparse


# Define the function to create the FastAPI project
def create_fastapi_project(project_name, endpoints):
    print("\033[36mCreating project structure...\033[0m")
    project_structure = {
        f"{project_name}/Dockerfile": """FROM python:3.12 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash build-essential git python3 python3-pip \
    && rm -rf /var/lib/apt/lists

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace

# Copy application code
COPY app /workspace/app
COPY utilities /workspace/utilities
COPY data /workspace/data

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    jupyter fastapi uvicorn


# Create repl.py during the build This will allow working directly with a repl on the right context uvicorn does
# use ipython repl.py
RUN echo \"import os\n\
import code\n\
\n\
# Ensure project root is in PYTHONPATH\n\
project_root = os.path.dirname(os.path.abspath(__file__))\n\
if project_root not in os.sys.path:\n\
    os.sys.path.insert(0, project_root)\n\
\n\
print('Interactive REPL loaded with project modules...')\n\
code.interact(local=dict(globals(), **locals()))\" > repl.py

# Expose FastAPI port
EXPOSE 8000

        
# Add Docker HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Set the default command to run the FastAPI app
CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\", \"--reload\"]
""",
        f"{project_name}/README.md": f"""# FastAPI Project

This is a dynamically generated FastAPI project. The API has the following endpoints:
{chr(10).join(f'- {endpoint["method"]}: `/{endpoint["name"]}`' for endpoint in endpoints)}

## Usage

To run the application locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To build and run the Docker container:

```bash
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app
```

To test the API in the container, run:

```bash
curl -X POST http://localhost:8000/{endpoints[0]["name"]} -H "Content-Type: application/json" -d '{{"text": "example"}}'
curl -X GET http://localhost:8000/{endpoints[1]["name"]}
```
""",
        f"{project_name}/app/__init__.py": "",
        f"{project_name}/app/main.py": f"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utilities.utils import some_utility_function

app = FastAPI()

@app.get("/health", tags=["Health"])
def health():
    return dict(status="ok")

        

        
class TextInput(BaseModel):
    text: str

{chr(10).join(create_endpoint_code(endpoint) for endpoint in endpoints)}
""",
        f"{project_name}/utilities/__init__.py": "",
        f"{project_name}/utilities/utils.py": "def some_utility_function():\n    pass\n",
        f"{project_name}/data/.gitkeep": "",
    }

    for filepath, content in project_structure.items():
        directory = os.path.dirname(filepath)
        if directory:  # Only create directories if the directory part is not empty
            os.makedirs(directory, exist_ok=True)
            print(f"\033[32mCreated directory: {directory}\033[0m")
        with open(filepath, "w") as f:
            f.write(content)
            print(f"\033[32mCreated file: {filepath}\033[0m")

    print("\033[35mProject structure successfully created.\033[0m")


def create_endpoint_code(endpoint):
    if endpoint["method"] == "POST":
        return f"@app.post(\"/{endpoint['name']}\")\nasync def {endpoint['name']}(input: TextInput):\n    try:\n        # Example POST logic\n        return {{\"message\": \"Processing input\", \"input\": input.text}}\n    except Exception as e:\n        raise HTTPException(status_code=500, detail=str(e))"
    elif endpoint["method"] == "GET":
        return f"@app.get(\"/{endpoint['name']}\")\nasync def {endpoint['name']}():\n    return {{\"message\": \"This is a stub for the GET endpoint: {endpoint['name']}\"}}"


if __name__ == "__main__":
    print("\033[36mParsing arguments...\033[0m")
    parser = argparse.ArgumentParser(
        description="Generate a FastAPI project dynamically."
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the project directory."
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        required=True,
        nargs="+",
        help="List of endpoints in the format 'name:method'.",
    )
    args = parser.parse_args()
    print("\033[32mArguments parsed successfully.\033[0m")

    parsed_endpoints = [
        {"name": ep.split(":")[0], "method": ep.split(":")[1].upper()}
        for ep in args.endpoints
    ]

    print(
        f"\033[36mCreating project: {args.name} with endpoints: {parsed_endpoints}\033[0m"
    )
    create_fastapi_project(args.name, parsed_endpoints)
    print("\033[32mProject creation complete!\033[0m")


# notes:

# def check_db():
#     try:
#         conn = sqlite3.connect("database.db")
#         conn.execute("SELECT 1")  # Simple query to check if DB is responsive
#         conn.close()
#         return True
#     except Exception:
#         return False

# @app.get("/health", tags=["Health"])
# def health():
#     db_status = "ok" if check_db() else "down"
#     return {
#         "status": "ok" if db_status == "ok" else "degraded",
#         "dependencies": {
#             "database": db_status
#         }
#     }
