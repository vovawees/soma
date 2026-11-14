#!/usr/bin/env bash
# Build the SOMA NeoForge mod with plain javac against the server's own classpath.
# No Gradle, no ModDevGradle - the server already has the patched MC classes.
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC_DIR="${MC_DIR:-/home/opc/ai_workspace/create-aeronautics-server}"
OUT="$ROOT/build"
CP_FILE="$OUT/cp.txt"
JAR="$OUT/soma-0.1.0.jar"

mkdir -p "$OUT/classes"

echo "[build] collecting classpath from $MC_DIR"
{
  ls "$MC_DIR"/libraries/net/neoforged/neoforge/21.1.265/*.jar
  find "$MC_DIR"/libraries -name '*.jar' ! -name '*-sources.jar' ! -name '*-javadoc.jar' \
       ! -path '*net/minecraft/server/1.21.1*'
  ls "$MC_DIR"/libraries/net/minecraft/server/1.21.1-20240808.144430/*-srg.jar
} | sort -u | tr '\n' ':' > "$CP_FILE"
echo "[build] classpath entries: $(tr ':' '\n' < "$CP_FILE" | grep -c .)"

echo "[build] compiling"
javac -nowarn -proc:none --release 21 \
  -cp "$(cat "$CP_FILE")" \
  -d "$OUT/classes" \
  $(find "$ROOT/src/main/java" -name '*.java')

echo "[build] packaging"
cp -r "$ROOT/src/main/resources/." "$OUT/classes/"
(cd "$OUT/classes" && jar --create --file "$JAR" .)

echo "[build] done -> $JAR ($(du -h "$JAR" | cut -f1))"