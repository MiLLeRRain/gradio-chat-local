@echo off

:: Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker is not installed. Please install Docker first.
    exit /b 1
)

:: Check if Docker Compose is installed
where docker-compose >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Docker Compose is not installed. Please install Docker Compose first.
    exit /b 1
)

:: Check if Node.js is installed (required for GitHub Copilot Language Server)
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Node.js is not installed. GitHub Copilot integration may not work.
    echo Please install Node.js to use GitHub Copilot features.
)

:: Check if NVIDIA Docker runtime is available
docker info | findstr /i nvidia >nul
if %ERRORLEVEL% neq 0 (
    echo Warning: NVIDIA Docker runtime not detected. GPU acceleration may not work.
    echo Please ensure nvidia-container-toolkit is installed and configured.
    set /p REPLY=Continue anyway? (y/n): 
    if /i not "%REPLY%"=="y" exit /b 1
)

:: Create models directory if it doesn't exist
if not exist .\models mkdir .\models
if not exist .\data mkdir .\data

echo Starting LLM Chat Application with Docker Compose...

:: Store the last build time if image exists
set LAST_BUILD_FILE=.last_build_time

:: Check if the image already exists
for /f "tokens=2" %%i in ('findstr /i "image:" docker-compose.yml') do set IMAGE_NAME=%%i
for /f "tokens=1" %%i in ('findstr /i "services:" docker-compose.yml /n ^| findstr /i /v "#" ^| for /f "tokens=1 delims=:" %%n in (\'findstr /i "services:" docker-compose.yml /n ^| findstr /i /v "#"\') do findstr /n "^" docker-compose.yml ^| findstr /b "%%n:" ^| for /f "tokens=2 delims=:" %%l in (\'more\') do echo %%l') do set SERVICE_NAME=%%i

:: Get current modification times
for /f "tokens=1" %%i in ('forfiles /p . /m app.py /c "cmd /c echo @ftime"') do set APP_MOD_TIME=%%i
for /f "tokens=1" %%i in ('forfiles /p . /m Dockerfile /c "cmd /c echo @ftime"') do set DOCKER_MOD_TIME=%%i
for /f "tokens=1" %%i in ('forfiles /p . /m requirements.txt /c "cmd /c echo @ftime"') do set REQ_MOD_TIME=%%i

:: Read last build time
set LAST_BUILD_TIME=0
if exist %LAST_BUILD_FILE% set /p LAST_BUILD_TIME=<%LAST_BUILD_FILE%

:: Check if any key files were modified since last build
set NEEDS_REBUILD=false
if %APP_MOD_TIME% gtr %LAST_BUILD_TIME% set NEEDS_REBUILD=true
if %DOCKER_MOD_TIME% gtr %LAST_BUILD_TIME% set NEEDS_REBUILD=true
if %REQ_MOD_TIME% gtr %LAST_BUILD_TIME% set NEEDS_REBUILD=true

:: Determine if we need to build
set BUILD_NEEDED=false
if "%IMAGE_NAME%"=="" (
    if not "%SERVICE_NAME%"=="" (
        for /f %%i in ('docker images ^| findstr "gradio-chat-local_%SERVICE_NAME%" ^| find /c /v ""') do set IMAGE_EXISTS=%%i
        if %IMAGE_EXISTS%==0 set BUILD_NEEDED=true
        if "%NEEDS_REBUILD%"=="true" set BUILD_NEEDED=true
    ) else (
        set BUILD_NEEDED=true
    )
) else (
    for /f %%i in ('docker images ^| findstr "%IMAGE_NAME%" ^| find /c /v ""') do set IMAGE_EXISTS=%%i
    if %IMAGE_EXISTS%==0 set BUILD_NEEDED=true
    if "%NEEDS_REBUILD%"=="true" set BUILD_NEEDED=true
)

:: Build and start the container
if "%BUILD_NEEDED%"=="true" (
    echo Building image (detected changes in application files)...
    docker-compose up --build
    :: Store current time as last build time
    echo %TIME% > %LAST_BUILD_FILE%
) else (
    echo No changes detected in application files. Starting without rebuild...
    docker-compose up
)