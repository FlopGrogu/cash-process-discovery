#!/bin/sh
set -eu

jar_path="${SPLIT_MINER_JAR:-/inputs/split-miner-1.7.1-all.jar}"
expected="472c006623d99a6e440aa93a58e29b867cc331cec2b12b3d7fb61fb2a5de8328"

if [ ! -f "$jar_path" ]; then
  echo "Split Miner JAR is not mounted at $jar_path" >&2
  exit 1
fi
actual="$(sha256sum "$jar_path" | cut -d ' ' -f 1)"
if [ "$actual" != "$expected" ]; then
  echo "Split Miner JAR SHA-256 mismatch: $actual" >&2
  exit 1
fi
java -version
echo "Split Miner 1.7.1 JAR verified."
