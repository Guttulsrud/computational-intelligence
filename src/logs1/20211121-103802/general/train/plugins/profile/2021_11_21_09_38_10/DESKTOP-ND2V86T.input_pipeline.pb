	????`?@????`?@!????`?@	L?RU,?S@L?RU,?S@!L?RU,?S@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????`?@ݔ?Z	]??1?w?}?a@I$??????Ys?SrN??@r0*	???()? A2O
Iterator::Root::Prefetch	3m????@!???T??X@)	3m????@1???T??X@:Preprocessing2E
Iterator::Root???#???@!      Y@)jj?Z_$??1A
F|?]?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 79.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9L?RU,?S@I?\\??a??Q^u???y4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ݔ?Z	]??ݔ?Z	]??!ݔ?Z	]??      ??!       "	?w?}?a@?w?}?a@!?w?}?a@*      ??!       2      ??!       :	$??????$??????!$??????B      ??!       J	s?SrN??@s?SrN??@!s?SrN??@R      ??!       Z	s?SrN??@s?SrN??@!s?SrN??@b      ??!       JGPUYL?RU,?S@b q?\\??a??y^u???y4@