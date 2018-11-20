CHECKPOINT_FILE=/home/undead/reps/ICNetUB/snapshots/model.ckpt-0
TF_PATH=/home/undead/reps/tensorflow
NAME=autovision_bisenet
export CUDA_VISIBLE_DEVICES=""
CUR_PATH=${PWD}

##########################################################################################

# Create unfrozen graph with export_inference_graph.py
python3 export_inference_graph.py --output_file ${NAME}_unfrozen.pb

python3 ${TF_PATH}/tensorflow/python/tools/freeze_graph.py \
--output_node_names="indices,label_names,label_colours,input_size,output_name" \
--input_graph=${NAME}_unfrozen.pb \
--input_checkpoint=${CHECKPOINT_FILE} \
--input_binary=true --output_graph=${NAME}_frozen.pb

#fold_old_batch_norms
${TF_PATH}/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=${CUR_PATH}/${NAME}_frozen.pb \
  --out_graph=${CUR_PATH}/${NAME}_cleaned.pb \
  --inputs="input"\
  --outputs="indices,label_names,label_colours,input_size,output_name"\
  --transforms='add_default_attributes
              strip_unused_nodes
              remove_nodes(op=Identity, op=CheckNumerics)
              fold_constants(ignore_errors=true)
              fold_batch_norms
              sort_by_execution_order'
