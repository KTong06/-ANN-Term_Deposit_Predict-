	?q???????q??????!?q??????	f?????	@f?????	@!f?????	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?q???????	?c??A c?ZB>??Yd?]K???*	gffff?R@2F
Iterator::Model??y?):??!???b??G@)6?;Nё??1=??<?sB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ׁsF??!?0?0:@)???{????1??䣓?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?!??u???!j??i??2@)/?$???1r?q?+@:Preprocessing2U
Iterator::Model::ParallelMapV2? ?	??!?%???^$@)? ?	??1?%???^$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{?G?z??!G'?|tJ@)?J?4q?1??8??8@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??H?}m?!?0?0@)??H?}m?1?0?0@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!Lh/???@)y?&1?l?1Lh/???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Mb??!?RJ?)5@)ŏ1w-!_?1???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9f?????	@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?	?c???	?c??!?	?c??      ??!       "      ??!       *      ??!       2	 c?ZB>?? c?ZB>??! c?ZB>??:      ??!       B      ??!       J	d?]K???d?]K???!d?]K???R      ??!       Z	d?]K???d?]K???!d?]K???JCPU_ONLYYf?????	@b 