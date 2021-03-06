?	?Z?kBI`@?Z?kBI`@!?Z?kBI`@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?Z?kBI`@???׭??1?0~??_@Iض(?A?@r0*	:??v?_@@2O
Iterator::Root::Prefetch*????!?sKI?M@)*????1?sKI?M@:Preprocessing2E
Iterator::Root ??XĠ?!      Y@)?*??O8??16????JD@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI????l?@Q?s???cX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???׭?????׭??!???׭??      ??!       "	?0~??_@?0~??_@!?0~??_@*      ??!       2      ??!       :	ض(?A?@ض(?A?@!ض(?A?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????l?@y?s???cX@?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2DUY??p???!UY??p???"-
IteratorGetNext/_2_Recv@vU?2 ??!?g(??u??"E
'sequential/xception/block1_conv2/Conv2DConv2D2(Sf?à?!??Q??׾?0"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative@@?HL??!?Q?W???"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2D>???h??!?yZ"??"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2DG????]??!?螌
.??"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2D0@Wt?\??!??)??9??"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx*34"B??!?Vp????"Q
5sequential/xception/block13_sepconv2/separable_conv2dConv2D?%??p9??!??Ɍ???"i
>sequential/xception/block3_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNativeh<?DӒ?!n?4????Q      Y@Y??)kʚ@a6eMYS?W@qaݬ^?W@y/(^??\?"?

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
Refer to the TF2 Profiler FAQb?94.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 