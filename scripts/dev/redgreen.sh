#!/bin/bash
# usage: redgreen.sh REPO SHA  -> prints RED and GREEN pytest summaries for the commit's test files
R=$1; C=$2; T=$(mktemp -d -t rg.XXXX)
git -C $R worktree add -q --detach $T $C^ >/dev/null
TESTS=$(git -C $R diff --name-only $C^ $C -- tests/ | grep -v '^tests/benchmarks/' | grep '\.py$')
(cd $T && git checkout -q $C -- $TESTS && PYTHONPATH=src:. timeout 1200 /opt/micromamba/envs/shiny/bin/pytest $TESTS -q -m "not bench" -p no:cacheprovider 2>&1 | tail -1 | sed "s/^/RED   /")
(cd $T && git checkout -q $C -- . && PYTHONPATH=src:. timeout 1200 /opt/micromamba/envs/shiny/bin/pytest $TESTS -q -m "not bench" -p no:cacheprovider 2>&1 | tail -1 | sed "s/^/GREEN /")
echo "TESTS $TESTS"
git -C $R worktree remove --force $T
