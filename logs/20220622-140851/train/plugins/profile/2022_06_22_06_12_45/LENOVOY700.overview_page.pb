?	6<?R???6<?R???!6<?R???	/????@/????@!/????@"e
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
	e?X???e?X???!e?X???      ??!       "      ??!       *      ??!       2	_)?Ǻ??_)?Ǻ??!_)?Ǻ??:      ??!       B      ??!       J	?0?*????0?*???!?0?*???R      ??!       Z	?0?*????0?*???!?0?*???JCPU_ONLYY.????@b Y      Y@q??h?a&U@"?
both?Your program is POTENTIALLY input-bound because 11.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.5997% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 