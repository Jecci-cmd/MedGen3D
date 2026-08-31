#!/usr/bin/env bash
set -u
repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/MedGen3D-main5task
main=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main
logs="$main/logs"
abdomen=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/data/AbdomenAtlas1.0Mini
reports=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-generation/datasets/CT-RATE-1500/metadata/dataset/radiology_text_reports
ctrate="$main/downloads/ctrate1300"
hf_home=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/huggingface
mkdir -p "$hf_home" "$hf_home/hub"
export HF_HOME="$hf_home"
export HF_HUB_CACHE="$hf_home/hub"
export HF_XET_CACHE="$hf_home/xet"
export HF_XET_HIGH_PERFORMANCE=1

alive() { [[ -s "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

# A ThreadPool download can remain alive indefinitely after all workers become
# stuck in network retries.  Track durable file-count progress rather than PID
# liveness alone so the watchdog can recover without deleting completed files.
ctrate_progress_file="$logs/ctrate1300_progress.state"
ctrate_stale_seconds=${CTRATE_STALE_SECONDS:-2700}

restart_stale_ctrate() {
  local count=$1 now last_count last_change pid
  now=$(date +%s)
  last_count=-1
  last_change=$now
  if [[ -s "$ctrate_progress_file" ]]; then
    read -r last_count last_change < "$ctrate_progress_file" || true
  fi
  if [[ "$count" -ne "$last_count" ]]; then
    printf '%s %s\n' "$count" "$now" > "$ctrate_progress_file"
    return
  fi
  if alive "$logs/ctrate1300_download.pid" && (( now - last_change >= ctrate_stale_seconds )); then
    pid=$(cat "$logs/ctrate1300_download.pid")
    printf '%s ctrate_stale count=%s unchanged_seconds=%s restarting_pid=%s\n' \
      "$(date -Is)" "$count" "$((now - last_change))" "$pid" >> "$logs/download_watchdog.log"
    kill "$pid" 2>/dev/null || true
    sleep 10
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$logs/ctrate1300_download.pid"
    printf '%s %s\n' "$count" "$now" > "$ctrate_progress_file"
  fi
}

while true; do
  abdomen_count=$(find "$abdomen/raw/extracted" -type f \( -name ct.nii.gz -o -name combined_labels.nii.gz \) | wc -l)
  if [[ "$abdomen_count" -lt 600 ]] && ! alive "$logs/abdomenatlas_1001_1300.pid"; then
    env HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=60 nohup /usr/bin/python \
      "$repo/scripts/download_abdomenatlas1000_direct.py" --first-case 1001 --last-case 1300 \
      --output-root "$abdomen" --workers 16 >> "$logs/abdomenatlas_1001_1300.log" 2>&1 &
    echo $! > "$logs/abdomenatlas_1001_1300.pid"
  fi
  ctrate_count=$(find "$ctrate/files" -type f -name '*.nii.gz' 2>/dev/null | wc -l)
  restart_stale_ctrate "$ctrate_count"
  if [[ "$ctrate_count" -lt 1300 ]] && ! alive "$logs/ctrate1300_download.pid"; then
    env HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=60 nohup /usr/bin/python \
      "$repo/scripts/download_ctrate_main.py" \
      --train-reports "$reports/train_reports.csv" \
      --valid-reports "$reports/validation_reports.csv" \
      --output-root "$ctrate" --workers 16 >> "$logs/ctrate1300_download.log" 2>&1 &
    echo $! > "$logs/ctrate1300_download.pid"
  fi
  printf '%s abdomen=%s/600 ctrate=%s/1300\n' "$(date -Is)" "$abdomen_count" "$ctrate_count" \
    >> "$logs/download_watchdog.log"
  sleep 120
done
