@echo off
REM Runs one daily update. Called by Windows Task Scheduler (see schedule_daily.bat): Mon-Fri 19:30, and also at
REM log-on and on wake from sleep so a missed 19:30 run is caught up. scripts\should_run.py skips the work when
REM the last successful run is newer than the most recent weekday 19:30 (pass --force to run anyway).
REM 1) Delivery Lens (Nifty 500 delivery scores)   2) IMPI live update (F&O model scores, signals, track records)
REM Every run appends start/end lines (with exit codes) to data\logs\nightly.log.
cd /d "%~dp0.."
if not exist data\logs mkdir data\logs
python scripts\should_run.py %* >> data\logs\nightly.log 2>&1
if errorlevel 1 exit /b 0
echo [%date% %time%] start >> data\logs\nightly.log
python -m src.update_daily >> data\logs\scheduler_stdout.txt 2>&1
set DL_RC=%ERRORLEVEL%
set IMPI_RC=skipped
if exist impi\.venv\Scripts\python.exe (
  pushd impi
  if not exist data\logs mkdir data\logs
  .venv\Scripts\python.exe -m impi.live >> data\logs\live_stdout.txt 2>&1
  call set IMPI_RC=%%ERRORLEVEL%%
  popd
)
echo [%date% %time%] end: delivery_lens=%DL_RC% impi_live=%IMPI_RC% >> data\logs\nightly.log
if "%DL_RC%"=="0" if "%IMPI_RC%"=="0" python -c "import datetime; print(datetime.datetime.now().isoformat(timespec='seconds'))" > data\logs\last_success.txt
if exist data\logs\update.lock del data\logs\update.lock
