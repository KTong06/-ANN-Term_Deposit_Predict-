	_?L???_?L???!_?L???	!?q?;P	@!?q?;P	@!!?q?;P	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$_?L???/n????AjM??St??Y?z?G???*	gffff?Q@2F
Iterator::Modele?X???!?ir?y)H@)???S㥛?1Ʋ????B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?o_???!??? ?R7@)y?&1???1???*??3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???Q???!1??? ?4@)?~j?t???1Z????0@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!?!??
?:%@)ŏ1w-!?1??
?:%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?l??????!+??o??I@)ŏ1w-!o?1??
?:@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?h?!Z????@)?~j?t?h?1Z????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!	???*@)??_vOf?1	???*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL7?A`???!????p7@)?~j?t?X?1Z???? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9"?q?;P	@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/n????/n????!/n????      ??!       "      ??!       *      ??!       2	jM??St??jM??St??!jM??St??:      ??!       B      ??!       J	?z?G????z?G???!?z?G???R      ??!       Z	?z?G????z?G???!?z?G???JCPU_ONLYY"?q?;P	@b 