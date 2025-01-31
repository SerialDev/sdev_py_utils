#!/bin/bash

PROJECT_NAME=$1
ACTION=$2

if [ -z "$PROJECT_NAME" ] || [ -z "$ACTION" ]; then
  echo -e "\033[31mUsage: $0 <project_name|all> <up|down|docker_cleanup|cleanup|dev>\033[0m"
  exit 1
fi

DOCKER_IMAGE="${PROJECT_NAME}-app"
DOCKER_CONTAINER="${PROJECT_NAME}-container"

pre_launch_check() {
  echo -e "\033[36mChecking prerequisites...\033[0m"

  if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
  fi

  if [ ! -d "./.venv" ]; then
    echo -e "\033[36mCreating a virtual environment (.venv)...\033[0m"
    uv venv
    echo -e "\033[32mVirtual environment created successfully.\033[0m"
  else
    echo -e "\033[33mVirtual environment already exists. Skipping creation.\033[0m"
  fi

  source .venv/bin/activate

  if [ "$VIRTUAL_ENV" != "$(pwd)/.venv" ]; then
    echo -e "\033[31mError: Failed to activate the correct virtual environment.\033[0m"
    exit 1
  fi

  if ! uv pip show uvicorn &> /dev/null; then
    echo -e "\033[36mInstalling Uvicorn...\033[0m"
    uv pip install uvicorn
    echo -e "\033[32mUvicorn installed successfully.\033[0m"
  else
    echo -e "\033[33mUvicorn is already installed. Skipping.\033[0m"
  fi

  if ! uv pip show fastapi &> /dev/null; then
    echo -e "\033[36mInstalling FastAPI...\033[0m"
    uv pip install fastapi
    echo -e "\033[32mFastAPI installed successfully.\033[0m"
  else
    echo -e "\033[33mFastAPI is already installed. Skipping.\033[0m"
  fi
}

handle_docker_compose() {
  if [ -f "docker-compose.yml" ]; then
    echo -e "\033[36mDetected docker-compose.yml, using Docker Compose...\033[0m"
    return 0
  else
    echo -e "\033[33mNo docker-compose.yml found, using standalone Docker commands.\033[0m"
    return 1
  fi
}

if [ "$PROJECT_NAME" == "all" ]; then
  case $ACTION in
    up)
      echo -e "\033[36mStarting all services using Docker Compose...\033[0m"
      docker-compose up --build -d
      echo -e "\033[32mAll services started successfully.\033[0m"
      exit 0
      ;;
    down)
      echo -e "\033[36mStopping all services...\033[0m"
      docker-compose down
      echo -e "\033[32mAll services stopped successfully.\033[0m"
      exit 0
      ;;
    *)
      echo -e "\033[31mInvalid action for 'all'. Use 'up' or 'down'.\033[0m"
      exit 1
      ;;
  esac
fi

if [ ! -d "./$PROJECT_NAME" ]; then
  echo -e "\033[31mError: Project directory '$PROJECT_NAME' does not exist.\033[0m"
  exit 1
fi

cd "./$PROJECT_NAME" || { echo -e "\033[31mFailed to change to project directory '$PROJECT_NAME'.\033[0m"; exit 1; }

case $ACTION in
  up)
    if handle_docker_compose; then
      echo -e "\033[36mStarting $PROJECT_NAME with Docker Compose...\033[0m"
      docker-compose up --build -d
    else
      echo -e "\033[36mBuilding the Docker image...\033[0m"
      docker build -t $DOCKER_IMAGE .
      echo -e "\033[32mDocker image built successfully.\033[0m"
      echo -e "\033[36mStarting the Docker container...\033[0m"
      docker run -d -p 8000:8000 --name $DOCKER_CONTAINER $DOCKER_IMAGE
    fi
    echo -e "\033[32m$PROJECT_NAME started successfully. Access the API at http://localhost:8000\033[0m"
    ;;
  down)
    if handle_docker_compose; then
      echo -e "\033[36mStopping $PROJECT_NAME with Docker Compose...\033[0m"
      docker-compose down
    else
      echo -e "\033[36mStopping and removing the Docker container...\033[0m"
      docker stop $DOCKER_CONTAINER && docker rm $DOCKER_CONTAINER
      echo -e "\033[32mDocker container stopped and removed successfully.\033[0m"
      echo -e "\033[36mRemoving the Docker image...\033[0m"
      docker rmi $DOCKER_IMAGE
      echo -e "\033[32mDocker image removed successfully.\033[0m"
    fi
    ;;
  docker_cleanup)
    echo -e "\033[36mCleaning up dangling Docker containers and images...\033[0m"
    docker container prune -f
    docker image prune -f
    echo -e "\033[32mDocker environment cleanup complete.\033[0m"
    ;;
  cleanup)
    echo -e "\033[36mRemoving the project directory...\033[0m"
    cd ..
    rm -rf "./$PROJECT_NAME"
    echo -e "\033[32mProject directory '$PROJECT_NAME' has been removed successfully.\033[0m"
    ;;
  dev)
    pre_launch_check
    echo -e "\033[36mLaunching the app in development mode with debug and reload enabled...\033[0m"

    if [ -n "$VIRTUAL_ENV" ]; then
      deactivate 2>/dev/null || true
    fi

    source .venv/bin/activate

    # Explicitly set the correct VIRTUAL_ENV path
    export VIRTUAL_ENV=$(pwd)/.venv

    # Verify the correct virtual environment is active
    if [ "$VIRTUAL_ENV" != "$(pwd)/.venv" ]; then
      echo -e "\033[31mError: Virtual environment mismatch.\033[0m"
      exit 1
    fi

    # Set the correct PYTHONPATH
    export PYTHONPATH=$(pwd)/app:$PYTHONPATH
    echo -e "\033[32mVirtual environment and PYTHONPATH configured successfully.\033[0m"
    echo -e "\033[36mStarting the development server...\033[0m"
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    echo -e "\033[32mDevelopment server is running successfully.\033[0m"
    ;;
  *)
    echo -e "\033[31mInvalid action. Use 'up', 'down', 'docker_cleanup', 'cleanup', or 'dev'.\033[0m"
    exit 1
    ;;
esac
