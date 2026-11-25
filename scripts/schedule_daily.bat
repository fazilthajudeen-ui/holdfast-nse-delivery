@echo off
REM Registers a Windows Task Scheduler job: Mon-Fri at 19:30 (set your PC clock to IST,
REM or change the time below to the local time that equals 19:30 IST).
REM Run once from the project folder:   scripts\schedule_daily.bat
REM Remove it later with:                schtasks /Delete /TN "NSE Delivery Update" /F
REM
REM The job runs in a HIDDEN window (closing a console window used to kill it: 9 Oct 2026, code 0xC000013A),
REM may run up to 3 hours, runs on battery, and catches up a missed 19:30 run: it also fires at log-on and
REM when the laptop wakes from sleep (a sleeping laptop missed the 12 Oct 2026 run). run_daily.bat asks
REM scripts\should_run.py first, so those extra triggers do nothing when the update is already done.
set TASK=NSE Delivery Update
powershell -NoProfile -Command ^
  "$a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -WindowStyle Hidden -Command \"& ''%~dp0run_daily.bat''\"') -WorkingDirectory '%~dp0..';" ^
  "$t1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 19:30;" ^
  "$t2 = New-ScheduledTaskTrigger -AtLogOn -User ($env:USERDOMAIN + '\' + $env:USERNAME); $t2.Delay = 'PT2M';" ^
  "$c = Get-CimClass -Namespace Root/Microsoft/Windows/TaskScheduler -ClassName MSFT_TaskEventTrigger;" ^
  "$t3 = New-CimInstance -CimClass $c -ClientOnly; $t3.Enabled = $true; $t3.Delay = 'PT2M';" ^
  "$t3.Subscription = '<QueryList><Query Id=\"0\" Path=\"System\"><Select Path=\"System\">*[System[Provider[@Name=''Microsoft-Windows-Power-Troubleshooter''] and EventID=1]]</Select></Query></QueryList>';" ^
  "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew;" ^
  "Register-ScheduledTask -TaskName '%TASK%' -Action $a -Trigger @($t1, $t2, $t3) -Settings $s -Force | Out-Null"
schtasks /Query /TN "%TASK%"
