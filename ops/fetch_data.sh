#!/usr/bin/env bash
# Download the three MaleCNS v1.0 feathers (~1.1 GB) and the FlyWire annotation table into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/flywire
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  [ -s "data/raw/$f" ] || curl -fL --progress-bar -o "data/raw/$f.part" "$B/$f" && mv -f "data/raw/$f.part" "data/raw/$f" 2>/dev/null || true
done
F=data/raw/flywire/Supplemental_file1_neuron_annotations.tsv
[ -s "$F" ] || curl -fL --progress-bar -o "$F" https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
ls -la data/raw data/raw/flywire
