	K????@K????@!K????@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0K????@?i???9(@1R?x?@A?ܚt["??I?tYLl?"@r0*	?????YD@2O
Iterator::Root::Prefetch??q????!dǵ??P@)??q????1dǵ??P@:Preprocessing2E
Iterator::Rootf??a?֤?!      Y@)lxz?,C??1?7q??@@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@??)?/??Q+EY?A?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?i???9(@?i???9(@!?i???9(@      ??!       "	R?x?@R?x?@!R?x?@*      ??!       2	?ܚt["???ܚt["??!?ܚt["??:	?tYLl?"@?tYLl?"@!?tYLl?"@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@??)?/??y+EY?A?X@