	??ݓ??????ݓ????!??ݓ????	{??T??@{??T??@!{??T??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ݓ?????MbX9??A?[ A?c??YvOjM??*	      U@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????K??!?N??N?J@)46<?R??1+D>?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0*??D??!`Y?K<@)/?$???1?HPS!?8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	?^)ː?!?"AM?h3@)???߾??1-?T?60@:Preprocessing2F
Iterator::ModelΈ?????!XV??6@) ?o_Ή?1???c??-@:Preprocessing2U
Iterator::Model::ParallelMapV2?~j?t?x?!?wɃg@)?~j?t?x?1?wɃg@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;pΈ?ް?!j*D>S@)????Mbp?1??WV?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOf?!M?h???	@)??_vOf?1M?h???	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!M?h???	@)??_vOf?1M?h???	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9{??T??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?MbX9???MbX9??!?MbX9??      ??!       "      ??!       *      ??!       2	?[ A?c???[ A?c??!?[ A?c??:      ??!       B      ??!       J	vOjM??vOjM??!vOjM??R      ??!       Z	vOjM??vOjM??!vOjM??JCPU_ONLYY{??T??@b 