#!/bin/sh -e

BASEDIR="$1"

cd $BASEDIR/depends;
make install URL=$2 PATCH=$3 DEST=$4
