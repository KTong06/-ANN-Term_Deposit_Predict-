	q???h ??q???h ??!q???h ??	L??"?-@L??"?-@!L??"?-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q???h ???C??????A+?????Y???H.??*	fffff&P@2F
Iterator::ModelF%u???!,T??nD@);?O??n??1;??~ ?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?&S???!?TGb,<@)K?=?U??1??k(?7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?]K?=??!??????4@)??ׁsF??1?~?p?.@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4??!8??˕*@)?J?4??18??˕*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipjM????!???M
?M@){?G?zt?1?+T??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!?P^Cy@)_?Q?k?1?P^Cy@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!{?*n??@)?????g?1{?*n??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Q???!?`?E87@)_?Q?[?1?P^Cy@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9L??"?-@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?C???????C??????!?C??????      ??!       "      ??!       *      ??!       2	+?????+?????!+?????:      ??!       B      ??!       J	???H.?????H.??!???H.??R      ??!       Z	???H.?????H.??!???H.??JCPU_ONLYYL??"?-@b 