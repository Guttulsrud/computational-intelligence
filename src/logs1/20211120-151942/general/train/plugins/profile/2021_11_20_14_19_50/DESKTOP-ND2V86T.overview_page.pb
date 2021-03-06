?	;?/K??`@;?/K??`@!;?/K??`@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails';?/K??`@Q??֥??1?K?W`@IU????3??r0*	P??n?;@2O
Iterator::Root::Prefetch΍?	K<??!?+??^mL@)΍?	K<??1?+??^mL@:Preprocessing2E
Iterator::RootA(??h???!      Y@)?4?;???1?p??E@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI???'m} @Q?B|X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Q??֥??Q??֥??!Q??֥??      ??!       "	?K?W`@?K?W`@!?K?W`@*      ??!       2      ??!       :	U????3??U????3??!U????3??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???'m} @y?B|X@?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2D?? w玬?!?? w玬?"-
IteratorGetNext/_2_Recv??????!Ic??<???"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative?#/'????!???T???"E
'sequential/xception/block1_conv2/Conv2DConv2DQL?
??!??5J7'??0"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2DSC???!??!?r(oK??"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2D!???8???!ϜE?^??"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2D??8ח?!p??[?Y??"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx?????!F?\?????"i
>sequential/xception/block3_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative????\??!#9i?+.??"P
4sequential/xception/block4_sepconv1/separable_conv2dConv2D?vk`N??!??ow`^??Q      Y@YLh/???@a{	?%??W@q?? ???W@yi?4?7\?"?

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
Refer to the TF2 Profiler FAQb?95.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 