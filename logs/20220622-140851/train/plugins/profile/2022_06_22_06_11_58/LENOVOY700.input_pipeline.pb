	?q???r @?q???r @!?q???r @	Ң{V~M @Ң{V~M @!Ң{V~M @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?q???r @????o??Alxz?,C??Y??g??s??*	33333?Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-???!($?zD@)?&1???1???c+B@:Preprocessing2F
Iterator::Model?0?*??!L;k?=@)?!??u???1??N?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?!??u???!??N?3@)?5?;Nс?1?c+???(@:Preprocessing2U
Iterator::Model::ParallelMapV29??v??z?!{???\"@)9??v??z?1{???\"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipݵ?|г??!-1>eڹQ@)??_vOv?1???c+?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOv?!???c+?@)??_vOv?1???c+?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!????%?@)y?&1?l?1????%?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvq?-??!x???P6@)_?Q?[?1?S???5@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Ң{V~M @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????o??????o??!????o??      ??!       "      ??!       *      ??!       2	lxz?,C??lxz?,C??!lxz?,C??:      ??!       B      ??!       J	??g??s????g??s??!??g??s??R      ??!       Z	??g??s????g??s??!??g??s??JCPU_ONLYYҢ{V~M @b 