	???~?:@???~?:@!???~?:@	???j/@???j/@!???j/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???~?:@k?w??#??A???K7???Y$(~??k??*	fffffz@2F
Iterator::Model???ZӼ??!?^v?7V@)??{??P??1?en$??U@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ZӼ???!?ڸ??@)?o_???1?NR7? @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?j+??ݓ?!>?W^?@)??ǘ????1??Ch?@:Preprocessing2U
Iterator::Model::ParallelMapV2F%u?{?!`H?A?L??)F%u?{?1`H?A?L??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+??Χ?!vM??G&@)n??t?1?Tk?n???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vOn?!?`4#?]??)???_vOn?1?`4#?]??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6j?!M°	g???)-C??6j?1M°	g???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA??ǘ???!HgZIF@)??H?}]?1????s???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t21.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???j/@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	k?w??#??k?w??#??!k?w??#??      ??!       "      ??!       *      ??!       2	???K7??????K7???!???K7???:      ??!       B      ??!       J	$(~??k??$(~??k??!$(~??k??R      ??!       Z	$(~??k??$(~??k??!$(~??k??JCPU_ONLYY???j/@b 