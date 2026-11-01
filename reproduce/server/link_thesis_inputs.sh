#!/usr/bin/env bash
# Link the inputs Table 4 needs from the old ~/thesis folder into this repo, in the layout the paper now uses
# (ageing-population transcript in its own folder). Run from the repo root on the GPU server:
#   bash reproduce/server/link_thesis_inputs.sh
set -euo pipefail
T="${THESIS:-$HOME/thesis}"
P=data/assemblies/processed_files
( cd "$T" && sha256sum -c --quiet "$OLDPWD/reproduce/server/thesis_inputs.sha256" ) \
  && echo "All 47 inputs in $T match the laptop copies." \
  || { echo "Some files differ (listed above). Upload those from the laptop instead (see rewrite.md)."; exit 1; }
mkdir -p $P/8th_amendment $P/ageing_population outputs/paper_v2 models
ln -sfn "$T/$P/gendereq" $P/gendereq
ln -sfn "$T/$P/scotland" $P/scotland
ln -sf "$T/$P/8th_amendment/Transcript-of-Final-Weekend-Sat-Sun-1_speeches.csv" $P/8th_amendment/
ln -sf "$T/$P/8th_amendment/Transcript-of-Sunday-9-July-Voting-_speeches.csv" $P/ageing_population/
ln -sfn "$T" outputs/embeddings
ln -sfn "$T/qwen3_expert_contrastive_final" models/qwen3_expert_contrastive_final
ls -l $P/*/ | head -20
