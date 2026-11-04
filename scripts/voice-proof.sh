#!/bin/zsh
# Speaks a known phrase through the speakers, records it from the microphone and
# transcribes it through WhisperCLIProvider. Restores the volume and mute state.
set -u
OUT=${1:?usage: voice-proof.sh <work-dir>}; PHRASE=${2:-"Shikki voice test. The quick brown fox jumps over the lazy dog."}
VOL=$(osascript -e 'output volume of (get volume settings)'); MUTED=$(osascript -e 'output muted of (get volume settings)')
restore() { osascript -e "set volume output volume $VOL" -e "set volume output muted $MUTED"; }
trap restore EXIT
osascript -e 'set volume output muted false' -e 'set volume output volume 70'
ffmpeg -y -loglevel error -f avfoundation -i ':Micro MacBook Pro' -t 9 -ar 16000 -ac 1 "$OUT/mic.wav" &
REC=$!
sleep 1; say -r 150 "$PHRASE"; wait $REC
restore; trap - EXIT
ffmpeg -hide_banner -i "$OUT/mic.wav" -af volumedetect -f null - 2>&1 | grep -E 'max_volume'
AIKIT_VOICE_WAV="$OUT/mic.wav" AIKIT_WHISPER_CLI=$(command -v whisper-cli) \
AIKIT_WHISPER_MODEL="$HOME/.local/share/whisper-cpp/models/ggml-large-v3-turbo.bin" \
AIKIT_VOICE_EXPECTED="$PHRASE" timeout 170 kagami test --scope AIKitVoiceLive --strict
