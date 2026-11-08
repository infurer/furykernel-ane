# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# publish-releases [--check]
#
# Master's pipeline, its release job: the release tags vVERSION merged into
# HEAD (hackage.sh) are checked and released, BASE by BASE, oldest first.
# - A BASE whose own tag vBASE is there goes to Hackage's index from that
#   tag's commit, the pre-release it promoted, with its documentation; then
#   the BASE's tags go to GitHub, and its candidate is deleted.
# - A BASE with pre-releases only: the newest one's commit goes to Hackage as
#   the candidate of BASE, replacing the one before, with its documentation;
#   then the BASE's tags go to GitHub.
# Uploads come before tags, so a tag on GitHub means that its upload went
# through, and a version Hackage refuses never shows as a tag there. A run
# after a failure carries on where the last one stopped, and one with
# nothing new does nothing.
#
# A tag must be annotated, signed with the key whose fingerprint is
# RELEASE_SIGNER (the public key in the file RELEASE_SIGNING_KEY), and its
# BASE be the version of its commit's package.
#
# GITHUB_REMOTE is the GitHub repository, GITHUB_SSH_COMMAND the ssh command
# that reaches it with the deploy key, and HACKAGE_TOKEN an API token of a
# maintainer of the package on Hackage.
#
# --check, in any checkout: the tarballs that a release of HEAD would upload
# (packages.hackage, built from HEAD's files), short of the upload. CI runs
# it before every merge.

prog=publish-releases

usage() {
  echo "usage: publish-releases [--check]" >&2
}

check=0
while [ "$#" -gt 0 ]; do
  case "$1" in
  --check)
    check=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
  esac
done

root=$(git rev-parse --show-toplevel)
cd "$root"

work=$(mktemp -d)
cleanup() {
  # Only the agent of the script's own keyring, never the user's.
  if [ "${GNUPGHOME:-}" = "$work/gnupg" ]; then
    gpgconf --kill gpg-agent 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

# A directory with the files of commit $1.
tree_of() {
  local dir=$work/tree-$1
  if [ ! -d "$dir" ]; then
    mkdir "$dir"
    git archive "$1" | tar -x -C "$dir"
  fi
  echo "$dir"
}

if [ "$check" = 1 ]; then
  tarballs "$(tree_of "$(git rev-parse HEAD)")"
  echo "publish-releases: a release of HEAD would upload $(basename "$sdist") and $(basename "$docs")"
  exit 0
fi

: "${RELEASE_SIGNER:?the fingerprint of the release signing key}"
: "${RELEASE_SIGNING_KEY:?a file with the release signing key}"
: "${GITHUB_REMOTE:?the GitHub repository}"
github_ssh=${GITHUB_SSH_COMMAND:-ssh}

GNUPGHOME=$work/gnupg
export GNUPGHOME
mkdir -m 700 "$GNUPGHOME"
gpg --batch --quiet --import "$RELEASE_SIGNING_KEY"

# The runner fetches only the pipeline's ref.
git fetch --quiet origin '+refs/tags/v*:refs/tags/v*'
github_tags=$(git -c core.sshCommand="$github_ssh" ls-remote --tags "$GITHUB_REMOTE" 'v*' | cut -f2)

# Whether GitHub has tag $1.
on_github() {
  grep -Fqx "refs/tags/$1" <<<"$github_tags"
}

# Checks release tag $1: annotated, signed with the release signing key, and
# on a commit whose package has the tag's BASE.
check_tag() {
  local tag=$1 status parsed base tree cabal version
  if [ "$(git cat-file -t "$tag")" != tag ]; then
    echo "publish-releases: $tag is not an annotated tag" >&2
    exit 1
  fi
  if ! status=$(git verify-tag --raw "$tag" 2>&1 >/dev/null); then
    echo "publish-releases: $tag's signature doesn't verify with the release signing key:" >&2
    echo "$status" >&2
    exit 1
  fi
  # VALIDSIG: the key that signed, ..., and last its primary key.
  if ! awk -v signer="$RELEASE_SIGNER" '$2 == "VALIDSIG" && ($3 == signer || $NF == signer) { found = 1 } END { exit !found }' <<<"$status"; then
    echo "publish-releases: $tag is not signed with $RELEASE_SIGNER" >&2
    exit 1
  fi
  parsed=$(parse_version "${tag#v}")
  read -r base _ _ <<<"$parsed"
  tree=$(tree_of "$(git rev-parse "$tag^{commit}")")
  cabal=$(cabal_file "$tree")
  version=$(cabal_field "$cabal" version)
  if [ "$version" != "$base" ]; then
    echo "publish-releases: $tag is on a commit whose package is at $version" >&2
    exit 1
  fi
}

# Pushes those of release tags $@ that GitHub doesn't have to it.
push_tags() {
  local tag refs=()
  for tag in "$@"; do
    if ! on_github "$tag"; then
      refs+=("refs/tags/$tag")
    fi
  done
  if [ "${#refs[@]}" -gt 0 ]; then
    git -c core.sshCommand="$github_ssh" push --quiet "$GITHUB_REMOTE" "${refs[@]}"
    echo "publish-releases: pushed ${refs[*]#refs/tags/} to GitHub"
  fi
}

# Releases BASE $1, whose release tags are the versions $2..., oldest first.
release_base() {
  local base=$1 version tags=() final='' newest='' tree cabal name pkgid tag
  local indexed in_index=0 has_docs=0 has_candidate=0 everywhere=1
  shift
  for version in "$@"; do
    tags+=("v$version")
    if [ "$version" = "$base" ]; then
      final=$version
    else
      newest=$version
    fi
  done
  tree=$(tree_of "$(git rev-parse "v${final:-$newest}^{commit}")")
  cabal=$(cabal_file "$tree")
  name=$(cabal_field "$cabal" name)
  pkgid=$name-$base

  for tag in "${tags[@]}"; do
    if ! on_github "$tag"; then
      check_tag "$tag"
    fi
  done

  if [ -z "$final" ]; then
    if on_github "v$newest"; then
      push_tags "${tags[@]}"
      echo "publish-releases: v$newest is released already"
      return
    fi
    tarballs "$tree"
    upload_package "$sdist" candidate
    upload_docs "$docs" "$pkgid" candidate
    echo "publish-releases: uploaded v$newest to Hackage as the candidate of $pkgid"
    push_tags "${tags[@]}"
    return
  fi

  indexed=$(index_versions "$name")
  if grep -Fqx "$base" <<<"$indexed"; then
    in_index=1
    if on_hackage "package/$pkgid/docs/"; then
      has_docs=1
    fi
  fi
  if on_hackage "package/$pkgid/candidate"; then
    has_candidate=1
  fi
  for tag in "${tags[@]}"; do
    on_github "$tag" || everywhere=0
  done
  if [ "$in_index$has_docs$has_candidate$everywhere" = 1101 ]; then
    echo "publish-releases: $pkgid is released already"
    return
  fi
  if [ "$has_docs" = 0 ]; then
    tarballs "$tree"
    if [ "$in_index" = 0 ]; then
      upload_package "$sdist" index
      echo "publish-releases: uploaded $pkgid to Hackage"
    fi
    upload_docs "$docs" "$pkgid" index
    echo "publish-releases: uploaded the documentation of $pkgid to Hackage"
  fi
  push_tags "${tags[@]}"
  if [ "$has_candidate" = 1 ]; then
    delete_candidate "$pkgid"
    echo "publish-releases: deleted the candidate of $pkgid from Hackage"
  fi
}

versions=$(git tag --merged HEAD --list 'v*' | sed 's/^v//' | sort_versions)
if [ -z "$versions" ]; then
  echo "publish-releases: no release tags"
  exit 0
fi
# BASE by BASE, from arrays: what release_base runs never reads a list
# meant for the loop.
mapfile -t all <<<"$versions"
mapfile -t bases < <(for version in "${all[@]}"; do
  parse_version "$version" | cut -d ' ' -f 1
done | uniq)
for base in "${bases[@]}"; do
  of_base=()
  for version in "${all[@]}"; do
    if [ "$(parse_version "$version" | cut -d ' ' -f 1)" = "$base" ]; then
      of_base+=("$version")
    fi
  done
  release_base "$base" "${of_base[@]}"
done
