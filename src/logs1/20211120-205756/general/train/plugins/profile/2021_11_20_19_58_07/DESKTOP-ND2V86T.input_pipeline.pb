	j?0
??b@j?0
??b@!j?0
??b@	?U?	?K???U?	?K??!?U?	?K??"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0j?0
??b@P?Lۿ2??1?%?Yb@In2?????Y?ٲ|]??r0*	??????B@2O
Iterator::Root::Prefetch???????!/????.O@)???????1/????.O@:Preprocessing2E
Iterator::RootΈ?????!      Y@)y?&1???1?DM4?B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?U?	?K??I???????QW???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	P?Lۿ2??P?Lۿ2??!P?Lۿ2??      ??!       "	?%?Yb@?%?Yb@!?%?Yb@*      ??!       2      ??!       :	n2?????n2?????!n2?????B      ??!       J	?ٲ|]???ٲ|]??!?ٲ|]??R      ??!       Z	?ٲ|]???ٲ|]??!?ٲ|]??b      ??!       JGPUY?U?	?K??b q???????yW???X@