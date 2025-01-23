#!/bin/bash

# Bash script to build, run, and manage the FastAPI Docker container
PROJECT_NAME=$1
ACTION=$2

if [ -z "$PROJECT_NAME" ] || [ -z "$ACTION" ]; then
  echo -e "\033[31mUsage: $0 <project_name> <up|down|docker_cleanup|cleanup>\033[0m"
  exit 1
fi

DOCKER_IMAGE="${PROJECT_NAME}-app"
DOCKER_CONTAINER="${PROJECT_NAME}-container"

case $ACTION in
  up)
    echo -e "\033[36mBuilding the Docker image...\033[0m"
    docker build -t $DOCKER_IMAGE ./$PROJECT_NAME
    echo -e "\033[36mStarting the Docker container...\033[0m"
    docker run -d -p 8000:8000 --name $DOCKER_CONTAINER $DOCKER_IMAGE
    echo -e "\033[32mContainer is running. Access the API at http://localhost:8000\033[0m"
    ;;
  down)
    echo -e "\033[36mStopping and removing the Docker container...\033[0m"
    docker stop $DOCKER_CONTAINER && docker rm $DOCKER_CONTAINER
    echo -e "\033[36mRemoving the Docker image...\033[0m"
    docker rmi $DOCKER_IMAGE
    echo -e "\033[32mDocker container and image removed.\033[0m"
    ;;
  docker_cleanup)
    echo -e "\033[36mCleaning up dangling Docker containers and images...\033[0m"
    docker container prune -f
    docker image prune -f
    echo -e "\033[32mDocker environment cleanup complete.\033[0m"
    ;;
  cleanup)
    echo -e "\033[36mRemoving the project directory...\033[0m"
    rm -rf ./$PROJECT_NAME
    echo -e "\033[32mProject directory '$PROJECT_NAME' has been removed.\033[0m"
    ;;
  *)
    echo -e "\033[31mInvalid action. Use 'up' to build and run, 'down' to stop and remove, 'docker_cleanup' to clean up Docker resources, or 'cleanup' to remove the project folder.\033[0m"
    exit 1
    ;;
esac
