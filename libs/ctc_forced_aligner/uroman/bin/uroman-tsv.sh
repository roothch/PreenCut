#!/usr/bin/env bash
# Created by Thamme Gowda on June 17, 2019

DIR=$(dirname "${BASH_SOURCE[0]}")  # get the directory name
# DIR=$(realpath "${DIR}")    # resolve its full path if need be

if [[ $# -lt 1 || $# -gt 2 ]]; then
    >&2 echo "ERROR: invalid args"
    >&2 echo "Usage: <input.tsv> [<output.tsv>]"
    exit 2
fi

INP=$1
OUT=$2

CMD=$DIR/uroman.pl

function romanize(){
    paste <(cut -f1 $INP) <(cut -f2 $INP | $CMD)
}

if [[ -n $OUT ]]; then
    romanize > $OUT
else
    romanize
fi


