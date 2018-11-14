#!/usr/bin/env bash

LOGS=${@:-./sample_1layer_run/*.out.1_layers.run*}
echo "LOGS: $LOGS"
OUTFILE=./sample_1layer_run/summary.out
>${OUTFILE}

for LOG in ${LOGS} ; do
    grep "Namespace" ${LOG} | tee -a ${OUTFILE}
    echo | tee -a ${OUTFILE}

    grep "exploitability of" ${LOG} | tee -a ${OUTFILE}
    echo | tee -a ${OUTFILE}

    echo "________________________________________________________________" | tee -a ${OUTFILE}
    echo | tee -a ${OUTFILE}
done
