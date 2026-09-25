#!/bin/zsh
# usage: research/guard.sh <max_GB> <cmd...>   -- kills the command if its RSS exceeds max_GB
max_kb=$(( $1 * 1048576 )); shift
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid | tr -d ' '); [[ -n $rss && $rss -gt $max_kb ]] && { echo "GUARD: killed (RSS ${rss}KB)"; kill -9 $pid; }
  sleep 1
done
wait $pid
