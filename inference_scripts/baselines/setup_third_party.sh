#!/bin/bash
# Clone the upstream repositories the baselines run against at the commits in third_party.lock, or verify existing checkouts.
# They are used unmodified (no patches): the repo's own scripts in this directory are the only additions.
#
#   inference_scripts/baselines/setup_third_party.sh            # into $THIRD_PARTY_DIR (default ~/software)
#
# Not in any repository, so fetched by hand (see README.md): MutateEverything's weights models/finetuning_ptm_2.pt and the
# ME-format inputs under $MUTATE_EVERYTHING_DIR/data (prepare/build_*_me_data.py write them; MSAs are data), SPURS's
# weights (Hugging Face hub), and the Python environments (envs/).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${THIRD_PARTY_DIR:-$HOME/software}
mkdir -p "$ROOT"
status=0
while read -r name url sha; do
  case "$name" in ''|'#'*) continue;; esac
  case "$name" in
    ThermoMPNN-D)           dir=${THERMOMPNN_DIR:-$ROOT/$name};;
    MutateEverything-oct06) dir=${MUTATE_EVERYTHING_DIR:-$ROOT/$name};;
    SPURS)                  dir=${SPURS_DIR:-$ROOT/$name};;
    ProteinMPNN)            dir=${PROTEINMPNN_DIR:-$ROOT/$name};;
  esac
  if [ ! -d "$dir/.git" ]; then
    echo "cloning $name -> $dir"
    git clone --quiet "$url" "$dir"
  fi
  head=$(cd "$dir" && git rev-parse HEAD)
  if [ "$head" != "$sha" ]; then
    echo "$name: at ${head:0:8}, lock says ${sha:0:8}; checking out the locked commit"
    (cd "$dir" && git checkout --quiet "$sha")
  fi
  if ! (cd "$dir" && git diff --quiet HEAD); then
    echo "WARNING: $name has modified tracked files; the baselines are specified against the unmodified checkout" >&2
    status=1
  fi
  echo "ok  $name  ${sha:0:8}  $dir"
done < "$HERE/third_party.lock"
exit $status
