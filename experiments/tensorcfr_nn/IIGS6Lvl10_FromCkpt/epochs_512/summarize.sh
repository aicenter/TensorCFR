#!/usr/bin/env bash

LOGS=${@:-*.out.*_layers}
echo "LOGS: $LOGS"
OUTFILE=summary.out
>${OUTFILE}

cat ${LOGS} | grep "args" >>${OUTFILE}
echo >>${OUTFILE}

cat ${LOGS} | grep "testset" >>${OUTFILE}
echo >>${OUTFILE}

cat ${LOGS} | grep "exploitability of" >>${OUTFILE}
