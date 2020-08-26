EXPERIMENT_TYPE=$1
FILE=results/$EXPERIMENT_TYPE
GPU_OPTS="-np 2 -H localhost:2"

if [ -z "$2" ]
  then
    echo "No Horovod GPU opts supplied. Script will run on localhost on 2 GPUs"
else
    GPU_OPTS="$2"
    echo "Horovod GPU opts: $GPU_OPTS"
fi



# if [[ -d $FILE ]]
# then
#     read -r -p "override results in folder? [y/N] " response
#     if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
#     then
#         horovodrun -np 2 \
#         -H localhost:4 \
#         python hvd_train_baseline.py 1001 $EXPERIMENT_TYPE
#     else
#         echo Stopping process to avoid overriding data.
#     fi
# else
#     mkdir results/$EXPERIMENT_TYPE
#     horovodrun -np 2 \
#         -H localhost:4 \
#         python hvd_train_baseline.py 1001 $EXPERIMENT_TYPE
# fi

if [[ -d $FILE ]]
then
    read -r -p "override results in folder? [y/N] " response
    if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
    then
        echo Stopping process to avoid overriding data.
        exit 0
    fi
else
    mkdir results/$EXPERIMENT_TYPE
fi

horovodrun \
    $GPU_OPTS \
    python hvd_train_baseline.py 1001 $EXPERIMENT_TYPE