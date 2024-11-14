#!/bin/bash

# Colors
COMMENT="\033[33m"      # Yellow for commentary
SUCCESS="\033[32m"      # Green for successful completions
ERROR="\033[31m"        # Red for errors
INFO="\033[35m"         # Magenta for informative/useful notes
HEADING="\033[36m"      # Cyan for section headings
RESET="\033[0m"         # Reset to default color

# Build settings
OUTPUT="sdev_c_utils.so"
SOURCE="sdev_c_utils.c"
PYTHON_INCLUDE=$(python3-config --includes)
PYTHON_LDFLAGS=$(python3-config --ldflags)

# Extract library path and library name using awk
LIB_PATH=$(echo $PYTHON_LDFLAGS | awk '{for(i=1;i<=NF;i++) if($i ~ /^-L/) print $i}')
LIB_NAME=$(echo $PYTHON_LDFLAGS | awk '{for(i=1;i<=NF;i++) if($i ~ /^-lpython/) print $i}')

# Check and add missing components if necessary
if [ -z "$LIB_PATH" ]; then
    echo -e "${INFO}No library path (-L) found, adding manually for Homebrew setup.${RESET}"
    LIB_PATH="-L/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib"
fi

if [ -z "$LIB_NAME" ]; then
    echo -e "${INFO}No Python library (-lpython) found, adding manually.${RESET}"
    LIB_NAME="-lpython3.13"
fi

# Begin build process
echo -e "${HEADING}Starting build process for ${SOURCE}...${RESET}"

# Compile command with explicit LIB_PATH and LIB_NAME
echo -e "${COMMENT}Compiling ${SOURCE} into ${OUTPUT}...${RESET}"
gcc -Wall -shared -o $OUTPUT $SOURCE -I$PYTHON_INCLUDE $LIB_PATH $LIB_NAME -ldl -framework CoreFoundation

# Check if compilation succeeded
if [ $? -eq 0 ]; then
    echo -e "${SUCCESS}Build successful! Output file: ${OUTPUT}${RESET}"
else
    echo -e "${ERROR}Build failed! Please check for errors above.${RESET}"
    exit 1
fi

# Output directory and location confirmation
OUTPUT_PATH=$(pwd)/$OUTPUT
if [ -f "$OUTPUT_PATH" ]; then
    echo -e "${INFO}The output file is saved at: ${OUTPUT_PATH}${RESET}"
else
    echo -e "${ERROR}Output file not found. Something went wrong.${RESET}"
    exit 1
fi

# Verify initialization function exists
echo -e "${INFO}Checking for initialization function (PyInit_sdev_c_utils)...${RESET}"
nm $OUTPUT | grep -q "PyInit_sdev_c_utils"
if [ $? -eq 0 ]; then
    echo -e "${SUCCESS}Initialization function PyInit_sdev_c_utils found.${RESET}"
else
    echo -e "${ERROR}Initialization function PyInit_sdev_c_utils not found. Please verify module setup.${RESET}"
    exit 1
fi

# Verification for architecture compatibility
echo -e "${INFO}Verifying build architecture compatibility...${RESET}"
file $OUTPUT | grep -q "arm64"
if [ $? -eq 0 ]; then
    echo -e "${SUCCESS}Output architecture is compatible with arm64.${RESET}"
else
    echo -e "${ERROR}Output architecture is not compatible with arm64. Please investigate further.${RESET}"
    exit 1
fi

# Completion message
echo -e "${HEADING}Build process completed successfully.${RESET}"
