	X9??v???X9??v???!X9??v???	zn~??@zn~??@!zn~??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X9??v???J{?/L???Aj?t???Y?St$????*	?????YX@2F
Iterator::ModelQ?|a2??!^?6?@E@)7?[ A??1?T?W?LA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?
F%u??!?_&??:@)'???????1?H??5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;?O???!????,?7@)??~j?t??1?my?ց3@:Preprocessing2U
Iterator::Model::ParallelMapV2? ?	??!?K?F?@)? ?	??1?K?F?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy?&1???!??T?W?L@)?I+?v?16?BW?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!?\0?Vm@)????Mbp?1?\0?Vm@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?\0?Vm@)????Mbp?1?\0?Vm@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapݵ?|г??!WD? ?9@)?J?4a?1??e??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9zn~??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J{?/L???J{?/L???!J{?/L???      ??!       "      ??!       *      ??!       2	j?t???j?t???!j?t???:      ??!       B      ??!       J	?St$?????St$????!?St$????R      ??!       Z	?St$?????St$????!?St$????JCPU_ONLYYzn~??@b 