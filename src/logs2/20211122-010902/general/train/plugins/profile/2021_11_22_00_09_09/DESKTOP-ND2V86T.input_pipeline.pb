	#??L?{@#??L?{@!#??L?{@	?-?Q@?-?Q@!?-?Q@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0#??L?{@}?!8.???1W??ma`@I$&??[???Y???7zs@r0*	??"[jA2O
Iterator::Root::Prefetch
?R?B?s@!M,??X@)
?R?B?s@1M,??X@:Preprocessing2E
Iterator::Root?V??l?s@!      Y@)?@?Ρ??1n_Y???j?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 70.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?-?Q@I??U???Q)??7?=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	}?!8.???}?!8.???!}?!8.???      ??!       "	W??ma`@W??ma`@!W??ma`@*      ??!       2      ??!       :	$&??[???$&??[???!$&??[???B      ??!       J	???7zs@???7zs@!???7zs@R      ??!       Z	???7zs@???7zs@!???7zs@b      ??!       JGPUY?-?Q@b q??U???y)??7?=@