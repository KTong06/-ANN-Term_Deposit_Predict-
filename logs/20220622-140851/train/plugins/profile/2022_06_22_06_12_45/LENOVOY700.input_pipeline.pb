	6<?R???6<?R???!6<?R???	/????@/????@!/????@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$6<?R???e?X???A_)?Ǻ??Y?0?*???*	??????K@2F
Iterator::Model?e??a???!?V???C@)???_vO??1?]tc?:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!ݘ??V?>@)?ZӼ???1P?&!?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<?R??!??<}??3@)???_vO~?1?]tc?*@:Preprocessing2U
Iterator::Model::ParallelMapV2?<,Ԛ?}?!c?>ZMB*@)?<,Ԛ?}?1c?>ZMB*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???x?&??!q?l?: N@)	?^)?p?1?ZMB@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!?_?.@)y?&1?l?1?_?.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!7????$@)Ǻ???f?17????$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??+e??!tc?>ZM6@)?~j?t?X?1`?.?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9.????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?X???e?X???!e?X???      ??!       "      ??!       *      ??!       2	_)?Ǻ??_)?Ǻ??!_)?Ǻ??:      ??!       B      ??!       J	?0?*????0?*???!?0?*???R      ??!       Z	?0?*????0?*???!?0?*???JCPU_ONLYY.????@b 