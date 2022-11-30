#!/bin/sh

echo Setting up environment with OS type: $OSTYPE
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    export CONFIG=`dirname $SCRIPT`
    export WORKINGDIR=`dirname $CONFIG`
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SCRIPT="$( cd "$( dirname "$0" )" && pwd )"
    export WORKINGDIR=`dirname $SCRIPT`
    export CONFIG=$WORKINGDIR/etc
fi

export DATA=$WORKINGDIR/data
export FRONTEND=$WORKINGDIR/frontend
export MODEL=$WORKINGDIR/models

# Creates redundancy in python path when sourced out of integreated shell but makes sure works for external shell aswell
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTHONPATH=$PYTHONPATH:$WORKINGDIR/src
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    export PYTHONPATH=$PYTHONPATH; $WORKINGDIR\\src
fi
dotenv -f ${WORKINGDIR}/.env set WORKINGDIR ${WORKINGDIR} 
dotenv -f ${WORKINGDIR}/.env set CONFIG ${CONFIG} 
dotenv -f ${WORKINGDIR}/.env set DATA ${DATA} 
dotenv -f ${WORKINGDIR}/.env set MODEL ${MODEL} 
dotenv -f ${WORKINGDIR}/.env set PYTHONPATH ${PYTHONPATH} 
