?	??K?z?a@??K?z?a@!??K?z?a@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??K?z?a@Ϡ??K??1?~??΁a@I???'E@r0*	?Zd;E@2O
Iterator::Root::Prefetch???4????!?[???aJ@)???4????1?[???aJ@:Preprocessing2E
Iterator::Root?f׽??!      Y@)Z?????1]?{?G@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ;~n|w@Q(?D\X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ϡ??K??Ϡ??K??!Ϡ??K??      ??!       "	?~??΁a@?~??΁a@!?~??΁a@*      ??!       2      ??!       :	???'E@???'E@!???'E@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ;~n|w@y(?D\X@?"R
6sequential_1/xception/block4_sepconv2/separable_conv2dConv2DK7j?l???!K7j?l???"-
IteratorGetNext/_2_Recv	????ѥ?!?ۣ?,???"G
)sequential_1/xception/block1_conv2/Conv2DConv2D???'y??!?=?ƴf??0"k
@sequential_1/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative_??̙?!?8<?S???"S
7sequential_1/xception/block14_sepconv2/separable_conv2dConv2DͿ︠??!?0Z?g???"R
6sequential_1/xception/block4_sepconv1/separable_conv2dConv2DNn?N????!??*?X???"R
6sequential_1/xception/block3_sepconv2/separable_conv2dConv2D???N???!?J????"R
6sequential_1/xception/block2_sepconv2/separable_conv2dConv2D8?z᝖?!]n%z>???"`
9sequential_1/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx?g,?0???!?}?K???"S
7sequential_1/xception/block13_sepconv2/separable_conv2dConv2D*??|????!ݑ3?i??Q      Y@YLh/???@a{	?%??W@q#W?? ?X@y????R^?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?99.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 