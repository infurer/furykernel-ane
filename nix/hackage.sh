# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# Hackage, for release and publish-releases, which this file comes before
# (release.nix): versions and their order, the package of a tree, Hackage's
# index and candidates, the tarballs a release uploads, and the PVP's view of
# a change of the API.
#
# A release is vVERSION, VERSION being BASE or BASE-STAGE.N: BASE the
# package's version, A.B.C.D by the PVP, and STAGE alpha, beta or rc, in that
# order and all before BASE itself. Cabal versions are numbers only, so
# Hackage has no pre-releases: a pre-release goes to Hackage as the candidate
# of its BASE, a preview outside the index that the next pre-release
# replaces, and BASE itself to the index, for good.
#
# HACKAGE_URL is Hackage, https://hackage.haskell.org unless set. Uploads
# take HACKAGE_TOKEN, an API token of a maintainer of the package. A script
# sets prog, its name, and work, a directory of its own, before it calls
# these.

hackage_url=${HACKAGE_URL:-https://hackage.haskell.org}

# The URL of Hackage's path $1 for a read, with a query of its own that no
# cache in front of Hackage has seen: Hackage's CDN serves pages of
# documentation and tarballs of candidates from its cache for hours, after
# they changed too.
fresh_url() {
  echo "$hackage_url/$1?fresh=$(date +%s%N)"
}

# VERSION $1's parts, "BASE STAGE N", with "release 0" for BASE itself; fails
# if $1 is no version.
parse_version() {
  if [[ $1 =~ ^((0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){3})(-(alpha|beta|rc)\.([1-9][0-9]*))?$ ]]; then
    if [ -n "${BASH_REMATCH[6]}" ]; then
      echo "${BASH_REMATCH[1]} ${BASH_REMATCH[6]} ${BASH_REMATCH[7]}"
    else
      echo "${BASH_REMATCH[1]} release 0"
    fi
  else
    return 1
  fi
}

# The versions on stdin, one per line, oldest first: BASEs as Cabal orders
# them, each after its pre-releases. A line that is no version fails.
sort_versions() {
  local line parsed base stage n rank
  while IFS= read -r line; do
    if [ -z "$line" ]; then
      continue
    fi
    if ! parsed=$(parse_version "$line"); then
      echo "$prog: $line is not a version such as 1.2.3.4 or 1.2.3.4-alpha.1" >&2
      exit 1
    fi
    read -r base stage n <<<"$parsed"
    case "$stage" in
    alpha) rank=1 ;;
    beta) rank=2 ;;
    rc) rank=3 ;;
    *) rank=4 ;;
    esac
    echo "$base $rank $n $line"
  done | LC_ALL=C sort -t ' ' -k1,1V -k2,2n -k3,3n | cut -d ' ' -f 4
}

# Whether VERSION $1 comes before VERSION $2.
version_lt() {
  local first
  first=$(printf '%s\n%s\n' "$1" "$2" | sort_versions | head -n 1)
  [ "$1" != "$2" ] && [ "$first" = "$1" ]
}

# The PVP's name for the step from BASE $1 to the newer BASE $2: major when
# A.B grew, minor when C did, patch otherwise.
bump() {
  local -a old new
  IFS=. read -r -a old <<<"$1"
  IFS=. read -r -a new <<<"$2"
  if [ "${old[0]}.${old[1]}" != "${new[0]}.${new[1]}" ]; then
    echo major
  elif [ "${old[2]}" != "${new[2]}" ]; then
    echo minor
  else
    echo patch
  fi
}

# Bump $1's place in patch < minor < major.
bump_rank() {
  case "$1" in
  major) echo 3 ;;
  minor) echo 2 ;;
  *) echo 1 ;;
  esac
}

# The .cabal file at the root of tree $1, which has exactly one.
cabal_file() {
  local files
  files=$(find "$1" -maxdepth 1 -name '*.cabal')
  if [ -z "$files" ] || [ "$(wc -l <<<"$files")" != 1 ]; then
    echo "$prog: $1 doesn't have exactly one .cabal file at its root" >&2
    exit 1
  fi
  echo "$files"
}

# Field $2 (name, version) of the .cabal file $1, from its top level.
cabal_field() {
  awk -v field="$2" '/^[^ \t]/ && tolower($1) == field ":" { print $2; exit }' "$1"
}

# The versions of package $1 on Hackage's index, one per line.
index_versions() {
  local status
  if ! status=$(curl -sS -L -o "$work/versions.json" -w '%{http_code}' -H 'Accept: application/json' "$(fresh_url "package/$1")"); then
    echo "$prog: Hackage didn't answer for $1" >&2
    exit 1
  fi
  case "$status" in
  200) jq -r 'keys[]' "$work/versions.json" ;;
  404) ;;
  *)
    echo "$prog: Hackage answered $status for $1" >&2
    exit 1
    ;;
  esac
}

# Whether Hackage has the page $1, such as package/PKGID/candidate: 200 or
# 404, and anything else an error.
on_hackage() {
  local status
  if ! status=$(curl -sS -L -o /dev/null -w '%{http_code}' "$(fresh_url "$1")"); then
    echo "$prog: Hackage didn't answer for /$1" >&2
    exit 1
  fi
  case "$status" in
  200) return 0 ;;
  404) return 1 ;;
  *)
    echo "$prog: Hackage answered $status for /$1" >&2
    exit 1
    ;;
  esac
}

# Whether Hackage's candidate of package version $1 is commit $2's: every
# file of its source tarball, the .cabal file that lists the others among
# them, as the commit has it. Cabal versions don't tell pre-releases apart,
# their files do.
candidate_is() {
  local pkgid=$1 commit=$2 status file
  rm -rf "$work/candidate"
  mkdir "$work/candidate"
  if ! status=$(curl -sS -L -o "$work/candidate.tar.gz" -w '%{http_code}' "$(fresh_url "package/$pkgid/candidate/$pkgid.tar.gz")"); then
    echo "$prog: Hackage didn't answer for the candidate of $pkgid" >&2
    exit 1
  fi
  case "$status" in
  200) ;;
  404) return 1 ;;
  *)
    echo "$prog: Hackage answered $status for the candidate of $pkgid" >&2
    exit 1
    ;;
  esac
  tar -xzf "$work/candidate.tar.gz" -C "$work/candidate"
  if [ ! -d "$work/candidate/$pkgid" ]; then
    return 1
  fi
  while IFS= read -r -d '' file; do
    file=${file#"$work/candidate/$pkgid/"}
    if ! git cat-file -e "$commit:$file" 2>/dev/null ||
      ! git show "$commit:$file" | cmp -s - "$work/candidate/$pkgid/$file"; then
      return 1
    fi
  done < <(find "$work/candidate/$pkgid" -type f -print0)
}

# Sends a request with HACKAGE_TOKEN to Hackage's path $1, curl's arguments
# after it, and shows Hackage's answer, which must be a success: 2xx, or
# 303, Hackage's redirect after it did what it was asked. The answer shows
# without its HTML tags, each of its lines ended, the last too, which a page
# on one line doesn't end. The token reaches curl in a file, never on a
# command line.
hackage_send() {
  local path=$1 status
  shift
  : "${HACKAGE_TOKEN:?an API token of a maintainer of the package on Hackage}"
  if [ ! -f "$work/hackage-auth" ]; then
    (umask 077 && printf 'Authorization: X-ApiKey %s\n' "$HACKAGE_TOKEN" >"$work/hackage-auth")
  fi
  if ! status=$(curl -sS -o "$work/answer" -w '%{http_code}' -H @"$work/hackage-auth" "$@" "$hackage_url/$path"); then
    echo "$prog: Hackage didn't answer for /$path" >&2
    exit 1
  fi
  awk '{ gsub(/<[^>]*>/, "") } !/^[[:space:]]*$/ { print "  hackage: " $0 }' "$work/answer"
  case "$status" in
  2?? | 303) ;;
  *)
    echo "$prog: Hackage answered $status for /$path" >&2
    exit 1
    ;;
  esac
}

# Uploads the source tarball $1 to Hackage's index, or with $2 candidate as
# the candidate of its version.
upload_package() {
  if [ "$2" = candidate ]; then
    hackage_send packages/candidates/ -F "package=@$1"
  else
    hackage_send upload -F "package=@$1"
  fi
}

# Uploads the documentation tarball $1 of package version $2, to the index's
# version or with $3 candidate to its candidate.
upload_docs() {
  local path=package/$2/docs
  if [ "$3" = candidate ]; then
    path=package/$2/candidate/docs
  fi
  hackage_send "$path" -X PUT -H 'Content-Type: application/x-tar' -H 'Content-Encoding: gzip' --data-binary "@$1"
}

# Deletes the candidate of package version $1, if Hackage still has it.
delete_candidate() {
  if on_hackage "package/$1/candidate"; then
    hackage_send "package/$1/candidate" -X DELETE
  fi
}

# The tarballs that tree $1's packages.hackage builds, with the caller's nix:
# sets pkgid; sdist, the source, PKGID.tar.gz; and docs, the documentation,
# PKGID-docs.tar.gz, with the Hoogle listing NAME.txt.
tarballs() {
  local cabal name out listing
  cabal=$(cabal_file "$1")
  name=$(cabal_field "$cabal" name)
  pkgid=$name-$(cabal_field "$cabal" version)
  # The tree's real path: nix refuses a path: flake with a symlink in its
  # path, as macOS's /tmp is.
  out=$(nix build --no-link --print-out-paths --no-update-lock-file "path:$(realpath "$1")#hackage")
  sdist=$out/$pkgid.tar.gz
  docs=$out/$pkgid-docs.tar.gz
  if [ ! -f "$sdist" ] || [ ! -f "$docs" ]; then
    echo "$prog: $out lacks $pkgid.tar.gz or $pkgid-docs.tar.gz" >&2
    exit 1
  fi
  listing=$(tar -tzf "$docs")
  if ! grep -Fqx "$pkgid-docs/$name.txt" <<<"$listing"; then
    echo "$prog: $pkgid-docs.tar.gz lacks the Hoogle listing $pkgid-docs/$name.txt" >&2
    exit 1
  fi
}

# The declarations of the Hoogle listing $1, one per line, each after its
# module: what a package exports, with types, constructors, classes and
# fixities, and the modules themselves. Instances go without one, since
# where an instance is defined doesn't matter to its users.
declarations() {
  awk '
    /^--/ || /^@/ || /^[[:space:]]*$/ { next }
    /^module / { module = $2; print; next }
    /^instance / { print; next }
    { print module ": " $0 }
  ' "$1" | LC_ALL=C sort -u
}

# What the change of the API from Hoogle listing $1 to listing $2 needs by
# the PVP: major when a declaration is gone or changed, minor when one was
# only added, patch otherwise. The declarations gone (-) and added (+) go to
# stderr.
api_change() {
  local old new gone added
  old=$(declarations "$1")
  new=$(declarations "$2")
  gone=$(LC_ALL=C comm -23 <(printf '%s\n' "$old") <(printf '%s\n' "$new"))
  added=$(LC_ALL=C comm -13 <(printf '%s\n' "$old") <(printf '%s\n' "$new"))
  if [ -n "$gone" ]; then
    awk '{ print "  - " $0 }' <<<"$gone" >&2
  fi
  if [ -n "$added" ]; then
    awk '{ print "  + " $0 }' <<<"$added" >&2
  fi
  if [ -n "$gone" ]; then
    echo major
  elif [ -n "$added" ]; then
    echo minor
  else
    echo "  (no declaration gone or added)" >&2
    echo patch
  fi
}
