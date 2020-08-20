EXPERIMENT_TYPE=$1
FILE=results/$EXPERIMENT_TYPE
if [[ -d $FILE ]]
then
    read -r -p "override results in folder? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
    then
        horovodrun -np 2 \
        -H localhost:4 \
        python hvd_train_baseline.py 1001 $EXPERIMENT_TYPE
    else
        echo Stopping process to avoid overriding data.
    fi
else
    mkdir results/$EXPERIMENT_TYPE
    horovodrun -np 2 \
        -H localhost:4 \
        python hvd_train_baseline.py 1001 $EXPERIMENT_TYPE
fi

