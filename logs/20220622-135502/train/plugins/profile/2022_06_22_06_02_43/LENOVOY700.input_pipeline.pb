	Gr?????Gr?????!Gr?????	?aL?6@?aL?6@!?aL?6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Gr?????I??&??Aa??+e??Y?g??s???*	?????yU@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!??!?v#??A@)B>?٬???17??x@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatݵ?|г??!?[$rS8=@)???????1?~?;@:Preprocessing2F
Iterator::Model	?c???!	??{??>@)??_?L??1-????68@:Preprocessing2U
Iterator::Model::ParallelMapV2?I+?v?!l,v?@)?I+?v?1l,v?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2??%䃮?!>?!?XQ@)F%u?k?1N?????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?J?4a?!`m????@)?J?4a?1`m????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!?v#??@)ŏ1w-!_?1?v#??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?q??????!"
?{ )B@)-C??6J?1$E? V???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?aL?6@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I??&??I??&??!I??&??      ??!       "      ??!       *      ??!       2	a??+e??a??+e??!a??+e??:      ??!       B      ??!       J	?g??s????g??s???!?g??s???R      ??!       Z	?g??s????g??s???!?g??s???JCPU_ONLYY?aL?6@b 