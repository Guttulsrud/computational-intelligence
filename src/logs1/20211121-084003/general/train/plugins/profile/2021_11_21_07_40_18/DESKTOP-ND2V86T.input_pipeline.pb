	?'????p@?'????p@!?'????p@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?'????p@d:tz????1???R?p@A3?&c`]?I?<?)[@r0*	$??C?>@2O
Iterator::Root::Prefetch???jHܓ?!R!???}O@)???jHܓ?1R!???}O@:Preprocessing2E
Iterator::Root???[v???!      Y@)$&??[X??1??Cv+?B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI`??,@Q=(??oX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d:tz????d:tz????!d:tz????      ??!       "	???R?p@???R?p@!???R?p@*      ??!       2	3?&c`]?3?&c`]?!3?&c`]?:	?<?)[@?<?)[@!?<?)[@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`??,@y=(??oX@