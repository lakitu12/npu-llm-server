#!/usr/bin/env bash
# Memory forensics monitor for the MT6991 AOT compile (run detached via setsid).
# One line per 5s: timestamp, mem used/total, swap used, RSS sum of the
# ci_convert/apply_plugin process tree. Written for the GitHub runner where the
# step shell itself has been killed (143/137) four runs in a row and file-only
# evidence was lost; the step also tails this file to its console.
while :; do
  rss=$(ps -eo rss,args | grep -E 'ci_convert|apply_plugin|ci_memmon' | grep -v grep | awk '{s+=$1} END {print s+0}')
  read -r mem swap < <(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} /^SwapTotal/{st=$2} /^SwapFree/{sf=$2} END {printf "%d/%dMi %dMi", (t-a)/1024, t/1024, (st-sf)/1024}' /proc/meminfo)
  cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  cgs=$(cat /sys/fs/cgroup/memory.swap.current 2>/dev/null)
  echo "$(date -u +%H:%M:%S) mem=$mem swap_used=$swap TREERSS_KB=$rss CG=${cg:-NA} CGSWAP=${cgs:-NA}"
  sleep 5
done
