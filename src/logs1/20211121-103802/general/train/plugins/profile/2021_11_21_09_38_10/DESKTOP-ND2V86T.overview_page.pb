?	????`?@????`?@!????`?@	L?RU,?S@L?RU,?S@!L?RU,?S@"q
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
	ݔ?Z	]??ݔ?Z	]??!ݔ?Z	]??      ??!       "	?w?}?a@?w?}?a@!?w?}?a@*      ??!       2      ??!       :	$??????$??????!$??????B      ??!       J	s?SrN??@s?SrN??@!s?SrN??@R      ??!       Z	s?SrN??@s?SrN??@!s?SrN??@b      ??!       JGPUYL?RU,?S@b q?\\??a??y^u???y4@?"R
6sequential_2/xception/block4_sepconv2/separable_conv2dConv2D???Z?˨?!???Z?˨?"G
)sequential_2/xception/block1_conv2/Conv2DConv2D!i1H???!?LƄ???0"-
IteratorGetNext/_1_Send׭4l4??!أ,?=??"S
7sequential_2/xception/block14_sepconv2/separable_conv2dConv2D??ƌ%??!?C̯?'??"k
@sequential_2/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNativeS??R??!x}?6r??"S
7sequential_2/xception/block13_sepconv2/separable_conv2dConv2D??D?V??!J7??t??"k
@sequential_2/xception/block3_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative'?gp??!/rZ??b??"R
6sequential_2/xception/block2_sepconv2/separable_conv2dConv2D??~??S??!ME?L`-??"R
6sequential_2/xception/block3_sepconv2/separable_conv2dConv2D?	??U???!>O?????"R
6sequential_2/xception/block6_sepconv2/separable_conv2dConv2DQ??"&???!c?{??.??Q      Y@Ye??S??3@a?
?[T@q!Hf_u(@y?-)??r?"?

host?Your program is HIGHLY input-bound because 79.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?12.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 