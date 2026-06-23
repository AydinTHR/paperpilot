#!/bin/sh
# commit-msg hook: strip any AI or co-author attribution from the commit message.
#
# Why this exists: the Git history should read as if written by a human engineer.
# Claude Code can append a "Co-authored-by" or "Generated with" footer to commits.
# The primary control is turning that off in Claude Code settings, but this hook is
# the backstop that guarantees it regardless of how any tool is configured, so an
# attribution trailer never reaches the log.
#
# $1 is the path to the file holding the proposed commit message.

set -eu

msg_file="$1"
tmp_file="$(mktemp)"

# Drop attribution lines, case-insensitively (-i). The patterns are scoped to AI
# attribution only so ordinary prose is never touched. "|| true" keeps the hook from
# aborting when grep filters every remaining line (grep exits non-zero if it prints
# nothing).
grep -viE \
  -e '^[[:space:]]*co-authored-by:.*(claude|anthropic)' \
  -e 'generated with.*claude' \
  -e 'noreply@anthropic\.com' \
  "$msg_file" > "$tmp_file" || true

# Remove any trailing blank lines left where the trailer used to be, then write back.
awk 'NF{last=NR} {line[NR]=$0} END{for (i=1; i<=last; i++) print line[i]}' \
  "$tmp_file" > "$msg_file"

rm -f "$tmp_file"
