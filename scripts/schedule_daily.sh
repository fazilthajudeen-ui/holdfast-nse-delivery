#!/usr/bin/env sh
# Linux/macOS: install a cron job that runs the update Mon-Fri at 19:30 IST.
# Usage: sh scripts/schedule_daily.sh      (prints and installs the line)
# cron uses the machine's local time; CRON_TZ is honoured by cronie/most Linux crons.
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LINE="CRON_TZ=Asia/Kolkata
30 19 * * 1-5 cd \"$PROJECT\" && python3 -m src.update_daily >> data/logs/scheduler_stdout.txt 2>&1"
echo "Adding to crontab:"; echo "$LINE"
( crontab -l 2>/dev/null | grep -v "src.update_daily" | grep -v "^CRON_TZ=Asia/Kolkata$"; echo "$LINE" ) | crontab -
# If your cron ignores CRON_TZ and the machine is not on IST, use 14:00 UTC instead:  0 14 * * 1-5 ...
