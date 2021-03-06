?	?=yX?@?=yX?@!?=yX?@	?=?/??T@?=?/??T@!?=?/??T@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9?=yX?@??? ????1aũ?Ba@Aɯb??s?I?iT?$??YC ?8???@r0*	??n??0&A2O
Iterator::Root::PrefetchF(????@!G!???X@)F(????@1G!???X@:Preprocessing2E
Iterator::Root??????@!      Y@)?pY?? ??1?R???NY?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 84.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?=?/??T@I@pJ??/??Q???7?/@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??? ??????? ????!??? ????      ??!       "	aũ?Ba@aũ?Ba@!aũ?Ba@*      ??!       2	ɯb??s?ɯb??s?!ɯb??s?:	?iT?$???iT?$??!?iT?$??B      ??!       J	C ?8???@C ?8???@!C ?8???@R      ??!       Z	C ?8???@C ?8???@!C ?8???@b      ??!       JGPUY?=?/??T@b q@pJ??/??y???7?/@?"R
6sequential_3/xception/block4_sepconv2/separable_conv2dConv2D|RW??֬?!|RW??֬?"G
)sequential_3/xception/block1_conv2/Conv2DConv2D??4e7???!<Ft?a??0"-
IteratorGetNext/_1_Send?(?מ?!?K?n??"k
@sequential_3/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative???e?+??!n??+0???"R
6sequential_3/xception/block2_sepconv2/separable_conv2dConv2DY?}.9+??!?N?Q???"S
7sequential_3/xception/block14_sepconv2/separable_conv2dConv2DdmS?Q??!???? ??"R
6sequential_3/xception/block4_sepconv1/separable_conv2dConv2Dv<?J??!5???"???"R
6sequential_3/xception/block3_sepconv2/separable_conv2dConv2D??xpR??!%F?Np???"`
9sequential_3/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx??Go???!LS?b???"R
6sequential_3/xception/block7_sepconv1/separable_conv2dConv2D1t9-??!??dO6??Q      Y@Ye??S??3@a?
?[T@q??EA?&@y~?3?xt?"?

host?Your program is HIGHLY input-bound because 84.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 