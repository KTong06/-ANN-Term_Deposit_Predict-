	6<?R???6<?R???!6<?R???	???c?
@???c?
@!???c?
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$6<?R???V-???A?,C????Y??????*	????̌X@2F
Iterator::ModeljM??S??!???Ӿ6D@)ŏ1w-!??14??>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???~?:??!/5N??#@@)c?ZB>???1????9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?&S???!2?vT??2@)V-???1?N???-@:Preprocessing2U
Iterator::Model::ParallelMapV2Έ?????!z? =??"@)Έ?????1z? =??"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6z?!?Qi"?@)-C??6z?1?Qi"?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ʡE????!VQ,A?M@)a??+ey?1MVQ,A@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vOn?!???7$@)???_vOn?1???7$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?0?*??!߿??C?4@)a2U0*?c?1d?ΙK?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???c?
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V-???V-???!V-???      ??!       "      ??!       *      ??!       2	?,C?????,C????!?,C????:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????JCPU_ONLYY???c?
@b 