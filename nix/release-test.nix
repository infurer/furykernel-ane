# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# The release scripts against throwaway repositories and keys, a fake nix
# whose packages.hackage is a pair of small tarballs, and a fake Hackage on
# 127.0.0.1 that keeps what it is sent. release makes a pre-release of
# master's version as a tag on master, of a newer version on a release
# commit that changes only the version, and the release itself as a tag on
# the newest pre-release; it refuses what isn't a new version, a release
# without a candidate, and a step smaller than the PVP needs.
# publish-releases uploads a pre-release as the candidate and the release to
# the index, each with its documentation, carries on after a failure, and
# refuses tags that aren't annotated, signed with the release signing key
# and on their version. Nothing leaves the build.
{
  git,
  gnupg,
  jq,
  publish-releases,
  python3,
  release,
  runCommand,
  writeShellScriptBin,
  writeText,
}: let
  # The parts of Hackage that the scripts use, for one package: its index,
  # the documentation of its versions, its candidates. It logs every write.
  fakeHackage = writeText "fake-hackage.py" ''
    import http.server
    import io
    import json
    import os
    import re
    import sys
    import tarfile

    state = sys.argv[1]
    token = "X-ApiKey " + sys.argv[2]
    index = set()
    docs = {}
    candidates = {}


    def log(line, name="log"):
        with open(os.path.join(state, name), "a") as f:
            f.write(line + "\n")


    def version(pkgid):
        return pkgid.rsplit("-", 1)[1]


    class Hackage(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, status, body=b"", kind="text/plain"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Hackage's CDN would answer a read without a query of its own
            # from its cache, maybe stale.
            path, _, query = self.path.partition("?")
            if not query.startswith("fresh="):
                log("GET " + path, "cached")
            m = re.fullmatch(r"/package/([a-z-]+)", path)
            if m and self.headers.get("Accept") == "application/json":
                if not index:
                    return self.answer(404)
                body = json.dumps({v: "normal" for v in index}).encode()
                return self.answer(200, body, "application/json")
            m = re.fullmatch(r"/package/(.+)/docs/", path)
            if m:
                return self.answer(200 if m.group(1) in docs else 404)
            m = re.fullmatch(r"/package/(.+)/docs/(.+[.]txt)", path)
            if m and m.group(1) in docs:
                with tarfile.open(fileobj=io.BytesIO(docs[m.group(1)])) as t:
                    member = t.extractfile(m.group(1) + "-docs/" + m.group(2))
                    return self.answer(200, member.read())
            m = re.fullmatch(r"/package/(.+)/candidate", path)
            if m:
                return self.answer(200 if m.group(1) in candidates else 404)
            m = re.fullmatch(r"/package/(.+)/candidate/(.+)[.]tar[.]gz", path)
            if m and m.group(1) == m.group(2) and m.group(1) in candidates:
                return self.answer(200, candidates[m.group(1)], "application/x-gzip")
            self.answer(404)

        def payload(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Authorization") != token:
                log(self.command + " " + path + " unauthorized")
                self.answer(401)
                return None
            return data

        # The form's file named package: its name without .tar.gz, and its
        # bytes.
        def package(self, data):
            kind = re.fullmatch(r"multipart/form-data; boundary=(.+)", self.headers.get("Content-Type", ""))
            if not kind:
                return None, None
            for part in data.split(b"--" + kind.group(1).encode()):
                head, blank, body = part.partition(b"\r\n\r\n")
                m = re.search(rb'name="package"; filename="([^"]+)[.]tar[.]gz"', head)
                if blank and m:
                    return m.group(1).decode(), body[:-2]
            return None, None

        def do_POST(self):
            data = self.payload()
            if data is None:
                return
            pkgid, tarball = self.package(data)
            if pkgid is None:
                return self.answer(400, b"no package")
            if version(pkgid) in index:
                return self.answer(403, b"Package name and version already exist")
            if self.path == "/upload":
                index.add(version(pkgid))
            elif self.path == "/packages/candidates/":
                candidates[pkgid] = tarball
            else:
                return self.answer(404)
            log("POST " + self.path + " " + pkgid)
            self.answer(200)

        def do_PUT(self):
            data = self.payload()
            if data is None:
                return
            m = re.fullmatch(r"/package/(.+?)/(candidate/)?docs", self.path)
            gzipped = self.headers.get("Content-Encoding") == "gzip"
            if not m or self.headers.get("Content-Type") != "application/x-tar" or not gzipped:
                return self.answer(400, b"not a gzipped tar")
            pkgid = m.group(1)
            failing = os.path.join(state, "fail-docs")
            if os.path.exists(failing):
                os.remove(failing)
                log("PUT " + self.path + " failed")
                return self.answer(500, b"failed, as the test asked")
            if m.group(2):
                if pkgid not in candidates:
                    return self.answer(404)
            elif version(pkgid) in index:
                docs[pkgid] = data
            else:
                return self.answer(404)
            log("PUT " + self.path)
            # As Hackage answers: a page on one line, with no newline at its end.
            page = "<html><body>Successfully uploaded documentation for " + pkgid + "!</body></html>"
            self.answer(200, page.encode(), "text/html")

        def do_DELETE(self):
            if self.payload() is None:
                return
            m = re.fullmatch(r"/package/(.+)/candidate", self.path)
            if not m or m.group(1) not in candidates:
                return self.answer(404)
            del candidates[m.group(1)]
            log("DELETE " + self.path)
            self.send_response(303)
            self.send_header("Location", "/package/demo/candidates/")
            self.send_header("Content-Length", "0")
            self.end_headers()


    server = http.server.HTTPServer(("127.0.0.1", 0), Hackage)
    with open(os.path.join(state, "port.tmp"), "w") as f:
        f.write(str(server.server_port))
    os.rename(os.path.join(state, "port.tmp"), os.path.join(state, "port"))
    server.serve_forever()
  '';

  # nix build path:TREE#hackage: TREE's tarballs, its .cabal file in the
  # source and its api.txt as the documentation's Hoogle listing. It notes
  # every tree it builds.
  fakeNix = writeShellScriptBin "nix" ''
    set -eu
    tree=
    for arg in "$@"; do
      case "$arg" in
      path:*'#hackage') tree=''${arg#path:} tree=''${tree%#hackage} ;;
      esac
    done
    [ -n "$tree" ] || { echo "fake nix: $*" >&2; exit 1; }
    name=$(awk '/^name:/ { print $2 }' "$tree"/*.cabal)
    pkgid=$name-$(awk '/^version:/ { print $2 }' "$tree"/*.cabal)
    out=$(mktemp -d "$TMPDIR/hackage-out.XXXXXX")
    mkdir -p "$out/src/$pkgid" "$out/src/$pkgid-docs"
    cp "$tree"/*.cabal "$out/src/$pkgid/"
    cp "$tree/api.txt" "$out/src/$pkgid-docs/$name.txt"
    echo "<html></html>" > "$out/src/$pkgid-docs/index.html"
    tar -czf "$out/$pkgid.tar.gz" -C "$out/src" "$pkgid"
    tar -czf "$out/$pkgid-docs.tar.gz" -C "$out/src" "$pkgid-docs"
    echo "$pkgid" >> "$TMPDIR/nix-builds"
    echo "$out"
  '';
in
  runCommand "release-test" {
    nativeBuildInputs = [fakeNix git gnupg jq publish-releases python3 release];
    # The fake Hackage listens on the loopback interface.
    __darwinAllowLocalNetworking = true;
  } ''
    export HOME=$TMPDIR
    export GNUPGHOME=$TMPDIR/gnupg
    mkdir -m 700 "$GNUPGHOME"
    # Whatever the end, the fake Hackage and gpg's agent go with the test:
    # left running, they would hold the build's log open. A failed command
    # names its line.
    hackage_pid=
    finish() {
      if [ -n "$hackage_pid" ]; then
        kill "$hackage_pid" 2> /dev/null || true
      fi
      gpgconf --kill all 2> /dev/null || true
    }
    trap finish EXIT
    trap 'echo "release-test: line $LINENO failed: $BASH_COMMAND" >&2' ERR
    gpg --batch --quiet --passphrase "" --quick-gen-key "Release Test <release@example.invalid>" ed25519 sign never
    gpg --batch --quiet --passphrase "" --quick-gen-key "Someone Else <else@example.invalid>" ed25519 sign never
    fingerprint() {
      gpg --list-keys --with-colons "$1" | awk -F: '$1 == "fpr" { print $10; exit }'
    }
    signer=$(fingerprint release@example.invalid)
    other=$(fingerprint else@example.invalid)
    gpg --armor --export "$signer" > signer.asc

    git config --global user.name "Release Test"
    git config --global user.email release@example.invalid
    git config --global user.signingkey "$signer"
    git config --global init.defaultBranch master

    mkdir hackage
    python3 ${fakeHackage} "$TMPDIR/hackage" test-token > hackage/server.log 2>&1 &
    hackage_pid=$!
    for _ in $(seq 100); do
      [ -e hackage/port ] && break
      sleep 0.1
    done
    export HACKAGE_URL=http://127.0.0.1:$(cat hackage/port) HACKAGE_TOKEN=test-token
    touch hackage/log

    # expect STATUS COMMAND...: COMMAND exits with STATUS.
    expect() {
      want=$1
      shift
      got=0
      "$@" > out 2>&1 || got=$?
      if [ "$got" -ne "$want" ]; then
        cat out
        echo "expected exit status $want, got $got: $*" >&2
        exit 1
      fi
    }
    # shows TEXT: the last command's output has TEXT.
    shows() {
      if ! grep -Fq -- "$1" out; then
        cat out
        echo "expected the output to show: $1" >&2
        exit 1
      fi
    }
    # shows_line LINE: the last command's output has LINE, a whole line.
    shows_line() {
      if ! grep -Fqx -- "$1" out; then
        cat out
        echo "expected the output to show the line: $1" >&2
        exit 1
      fi
    }
    # logged COUNT LINE: Hackage logged LINE COUNT times.
    logged() {
      if [ "$(grep -Fxc -- "$2" "$TMPDIR/hackage/log")" != "$1" ]; then
        cat "$TMPDIR/hackage/log"
        echo "expected Hackage to have logged $1 times: $2" >&2
        exit 1
      fi
    }
    # on_github TAG: the GitHub repository has TAG.
    on_github() {
      git ls-remote --tags "$GITHUB_REMOTE" | cut -f2 | grep -Fqx "refs/tags/$1"
    }
    # commit MESSAGE: commits the demo's changes and pushes master.
    commit() {
      git commit --quiet -am "$1"
      git push --quiet origin master
    }

    # A package at 0.2.0.0: its .cabal file, cabal2nix's output, and what
    # stands for its Hoogle listing.
    git init --quiet --bare demo.git
    git init --quiet --bare github.git
    git init --quiet demo
    cd demo
    cat > demo.cabal <<'EOF'
    cabal-version: 3.0
    name: demo
    version: 0.2.0.0
    license: BSD-3-Clause

    library
      exposed-modules: Demo
      build-depends: base >=4.21 && <4.22
    EOF
    cat > demo.nix <<'EOF'
    { mkDerivation, base, lib }:
    mkDerivation {
      pname = "demo";
      version = "0.2.0.0";
      src = ./.;
      libraryHaskellDepends = [ base ];
      license = lib.licenses.bsd3;
    }
    EOF
    cat > api.txt <<'EOF'
    -- Hoogle documentation, generated by Haddock
    @package demo
    @version 0.2.0.0

    -- | The demo.
    module Demo

    -- | A greeting.
    greet :: String -> String
    EOF
    git add -A
    git commit --quiet -m "demo: init at 0.2.0.0"
    git remote add origin "$TMPDIR/demo.git"
    git push --quiet origin master
    export GITHUB_REMOTE=$TMPDIR/github.git RELEASE_SIGNER=$signer RELEASE_SIGNING_KEY=$TMPDIR/signer.asc

    # Not a version, a version older than master's, and a release that had
    # no pre-release.
    expect 2 release --no-push --yes 0.2.0
    expect 2 release --no-push --yes 0.2.0.0-pre.1
    expect 2 release --no-push --yes 0.2.0.0-alpha.0
    expect 1 release --no-push --yes 0.1.0.0-alpha.1
    shows "newer than 0.1.0.0"
    expect 1 release --no-push --yes 0.2.0.0
    shows "has had no pre-release"

    # A pre-release of master's version: a signed tag on master's head, no
    # commit and no branch, and no PVP check before anything is on Hackage.
    head=$(git rev-parse master)
    expect 0 release --no-push --yes 0.2.0.0-alpha.1
    [ "$(git rev-parse 'v0.2.0.0-alpha.1^{commit}')" = "$head" ]
    git verify-tag v0.2.0.0-alpha.1 2> /dev/null
    [ "$(git rev-parse master)" = "$head" ]
    [ -z "$(git branch --list 'release/*')" ]
    [ "$(git worktree list | wc -l)" = 1 ]
    [ ! -e "$TMPDIR/nix-builds" ]
    expect 1 release --no-push --yes 0.2.0.0-alpha.1
    shows "v0.2.0.0-alpha.1 exists already"
    git push --quiet origin v0.2.0.0-alpha.1

    # Master's pipeline makes it the candidate, then pushes its tag.
    expect 0 publish-releases
    shows_line "  hackage: Successfully uploaded documentation for demo-0.2.0.0!"
    shows_line "publish-releases: uploaded v0.2.0.0-alpha.1 to Hackage as the candidate of demo-0.2.0.0"
    logged 1 "POST /packages/candidates/ demo-0.2.0.0"
    logged 1 "PUT /package/demo-0.2.0.0/candidate/docs"
    on_github v0.2.0.0-alpha.1
    expect 0 publish-releases
    shows "v0.2.0.0-alpha.1 is released already"
    logged 1 "POST /packages/candidates/ demo-0.2.0.0"

    # A beta of a later master replaces the candidate. An alpha after it is
    # refused, and so is the release that promoted no candidate yet.
    echo "synopsis: A demo" >> demo.cabal
    commit "demo: synopsis"
    expect 0 release --no-push --yes 0.2.0.0-beta.1
    git push --quiet origin v0.2.0.0-beta.1
    expect 1 release --no-push --yes 0.2.0.0-alpha.2
    shows "v0.2.0.0-beta.1 is newer than 0.2.0.0-alpha.2"
    expect 1 release --no-push --yes 0.2.0.0
    shows "candidate of demo 0.2.0.0 isn't v0.2.0.0-beta.1 yet"
    expect 0 publish-releases
    logged 2 "POST /packages/candidates/ demo-0.2.0.0"
    on_github v0.2.0.0-beta.1

    # The release promotes the beta: its tag goes on the beta's commit, not
    # on master's newer head.
    beta=$(git rev-parse 'v0.2.0.0-beta.1^{commit}')
    echo "notes" > notes
    git add notes
    commit "demo: notes"
    expect 0 release --no-push --yes 0.2.0.0
    shows "promotes v0.2.0.0-beta.1"
    [ "$(git rev-parse 'v0.2.0.0^{commit}')" = "$beta" ]
    git verify-tag v0.2.0.0 2> /dev/null
    git push --quiet origin v0.2.0.0

    # Its documentation fails once: nothing reaches GitHub, and the next run
    # uploads the documentation alone, pushes the tag and deletes the
    # candidate.
    touch "$TMPDIR/hackage/fail-docs"
    expect 1 publish-releases
    logged 1 "POST /upload demo-0.2.0.0"
    logged 1 "PUT /package/demo-0.2.0.0/docs failed"
    if on_github v0.2.0.0; then
      echo "a release whose documentation failed reached GitHub" >&2
      exit 1
    fi
    expect 0 publish-releases
    logged 1 "POST /upload demo-0.2.0.0"
    logged 1 "PUT /package/demo-0.2.0.0/docs"
    logged 1 "DELETE /package/demo-0.2.0.0/candidate"
    on_github v0.2.0.0
    expect 0 publish-releases
    shows "demo-0.2.0.0 is released already"
    expect 1 release --no-push --yes 0.2.0.0-rc.1
    shows "demo 0.2.0.0 is on Hackage already"

    # A newer version's pre-release: a signed release commit that changes
    # only the version, and on it the tag; the PVP check finds the API of
    # 0.2.0.0 on Hackage unchanged.
    expect 0 release --no-push --yes 0.2.0.1-alpha.1
    shows "no declaration gone or added"
    [ "$(git log -1 --format=%s release/v0.2.0.1-alpha.1)" = "demo: 0.2.0.1" ]
    git log -1 --format=%b release/v0.2.0.1-alpha.1 | grep -Fqx "Signed-off-by: Release Test <release@example.invalid>"
    git verify-commit release/v0.2.0.1-alpha.1 2> /dev/null
    git verify-tag v0.2.0.1-alpha.1 2> /dev/null
    [ "$(git rev-parse 'v0.2.0.1-alpha.1^{commit}')" = "$(git rev-parse release/v0.2.0.1-alpha.1)" ]
    [ "$(git diff --name-only master release/v0.2.0.1-alpha.1 | tr '\n' ' ')" = "demo.cabal demo.nix " ]
    git show release/v0.2.0.1-alpha.1:demo.cabal | grep -Fqx "version: 0.2.0.1"
    git show release/v0.2.0.1-alpha.1:demo.nix | grep -Fqx '  version = "0.2.0.1";'
    [ "$(git worktree list | wc -l)" = 1 ]
    git tag -d v0.2.0.1-alpha.1 > /dev/null
    git branch -D release/v0.2.0.1-alpha.1 > /dev/null

    # By the PVP, an export added needs a new minor version, and one gone a
    # new major one, which --skip-pvp-check lets through.
    printf '\n-- | A farewell.\nfarewell :: String -> String\n' >> api.txt
    commit "Demo: farewell"
    expect 1 release --no-push --yes 0.2.0.1-alpha.1
    shows "+ Demo: farewell :: String -> String"
    shows "needs a new minor version"
    expect 0 release --no-push --yes 0.2.1.0-alpha.1
    git tag -d v0.2.1.0-alpha.1 > /dev/null
    git branch -D release/v0.2.1.0-alpha.1 > /dev/null
    sed -i '/^greet ::/d' api.txt
    commit "Demo: no greeting"
    expect 1 release --no-push --yes 0.2.1.0-alpha.1
    shows "- Demo: greet :: String -> String"
    shows "needs a new major version"
    expect 0 release --no-push --yes --skip-pvp-check 0.2.1.0-alpha.1
    git tag -d v0.2.1.0-alpha.1 > /dev/null
    git branch -D release/v0.2.1.0-alpha.1 > /dev/null
    expect 0 release --no-push --yes 0.3.0.0-alpha.1
    git tag -d v0.3.0.0-alpha.1 > /dev/null
    git branch -D release/v0.3.0.0-alpha.1 > /dev/null

    # Tags that must not release: lightweight, unsigned, signed with another
    # key, and on a commit of another version. None reaches Hackage or
    # GitHub.
    sed -i 's/^version: 0.2.0.0$/version: 0.4.0.0/' demo.cabal
    sed -i 's/^  version = "0.2.0.0";$/  version = "0.4.0.0";/' demo.nix
    commit "demo: 0.4.0.0"
    writes=$(wc -l < "$TMPDIR/hackage/log")
    git tag v0.4.0.0-alpha.1
    git push --quiet origin v0.4.0.0-alpha.1
    expect 1 publish-releases
    shows "v0.4.0.0-alpha.1 is not an annotated tag"
    git tag -f -a -m "demo 0.4.0.0-alpha.1" v0.4.0.0-alpha.1 > /dev/null
    git push --quiet --force origin v0.4.0.0-alpha.1
    expect 1 publish-releases
    shows "signature doesn't verify"
    git tag -f -s -u "$other" -m "demo 0.4.0.0-alpha.1" v0.4.0.0-alpha.1 > /dev/null
    git push --quiet --force origin v0.4.0.0-alpha.1
    expect 1 publish-releases
    shows "signature doesn't verify"
    git tag -d v0.4.0.0-alpha.1 > /dev/null
    git push --quiet origin --delete v0.4.0.0-alpha.1
    git tag -s -m "demo 0.5.0.0-alpha.1" v0.5.0.0-alpha.1
    git push --quiet origin v0.5.0.0-alpha.1
    expect 1 publish-releases
    shows "v0.5.0.0-alpha.1 is on a commit whose package is at 0.4.0.0"
    [ "$(wc -l < "$TMPDIR/hackage/log")" = "$writes" ]
    if on_github v0.4.0.0-alpha.1 || on_github v0.5.0.0-alpha.1; then
      echo "a refused tag reached GitHub" >&2
      exit 1
    fi
    git tag -d v0.5.0.0-alpha.1 > /dev/null
    git push --quiet origin --delete v0.5.0.0-alpha.1

    # What a release of HEAD would upload, without uploading it.
    expect 0 publish-releases --check
    shows "a release of HEAD would upload demo-0.4.0.0.tar.gz and demo-0.4.0.0-docs.tar.gz"
    [ "$(wc -l < "$TMPDIR/hackage/log")" = "$writes" ]

    if grep -F unauthorized "$TMPDIR/hackage/log"; then
      echo "a write reached Hackage without the token" >&2
      exit 1
    fi
    if [ -e "$TMPDIR/hackage/cached" ]; then
      cat "$TMPDIR/hackage/cached"
      echo "these reads could have been answered from a cache" >&2
      exit 1
    fi
    touch $out
  ''
