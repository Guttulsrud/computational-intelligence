	??up0?`@??up0?`@!??up0?`@	ٓi5f??ٓi5f??!ٓi5f??"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??up0?`@"?*??<??1?k?6??`@I?H?s
???YuWv?????r0*	^?I???A2O
Iterator::Root::PrefetchǞ=??v@!?@????X@)Ǟ=??v@1?@????X@:Preprocessing2E
Iterator::Root???
v@!      Y@)???????1?L?_??h?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ؓi5f??I????????Qo?z?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"?*??<??"?*??<??!"?*??<??      ??!       "	?k?6??`@?k?6??`@!?k?6??`@*      ??!       2      ??!       :	?H?s
????H?s
???!?H?s
???B      ??!       J	uWv?????uWv?????!uWv?????R      ??!       Z	uWv?????uWv?????!uWv?????b      ??!       JGPUYؓi5f??b q????????yo?z?X@