#!/bin/bash
VERSION="v01"
PROJECT="synpcg"
mode=${1:-none}
HOSTNAME_C=$PROJECT"-"`hostname`
CONTAINER_NAME=$PROJECT"-"$VERSION

if [ $mode = "build" ]; then
    ORIGINAL_DATA_PATH=$(yq e '.path.original_data' config.yaml)
    PROCESSED_DATA_PATH=$(yq e '.path.processed_data' config.yaml)
    DIRNAME=$(basename "`pwd`")
    CONTAINER_HOMEDIR="/home/$USER/$DIRNAME/"

    docker build \
        --build-arg USERNAME=$USER \
        --build-arg UID=$UID \
        --build-arg GID=$(id -g $USER) \
        --build-arg ORIGINAL_DATA_LOC=$ORIGINAL_DATA_PATH \
        --build-arg PROCESSED_DATA_PATH=$PROCESSED_DATA_PATH \
        -t $PROJECT:$VERSION . < Dockerfile
    docker run \
        --gpus all \
        -v $ORIGINAL_DATA_PATH:$ORIGINAL_DATA_PATH \
        -v $PROCESSED_DATA_PATH:$PROCESSED_DATA_PATH \
        -v $(pwd):$CONTAINER_HOMEDIR \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash
elif [ $mode = "start" ]; then
    docker run \
        --gpus all \
        -v $ORIGINAL_DATA_PATH:$ORIGINAL_DATA_PATH \
        -v $PROCESSED_DATA_PATH:$PROCESSED_DATA_PATH \
        -v $(pwd):$CONTAINER_HOMEDIR \
        -it -d --shm-size=180g \
        --hostname=$HOSTNAME_C \
        --name $CONTAINER_NAME $PROJECT:$VERSION  /bin/bash
elif [ $mode = "restart" ]; then
    docker start $CONTAINER_NAME 
fi

docker exec -it $CONTAINER_NAME /bin/bash
