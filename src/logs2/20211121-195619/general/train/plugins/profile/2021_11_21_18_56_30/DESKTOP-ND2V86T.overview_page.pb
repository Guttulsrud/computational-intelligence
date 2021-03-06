?	?X??+?`@?X??+?`@!?X??+?`@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?X??+?`@d@?z?G??1?? ?y`@I????1@r0*	??????>@2O
Iterator::Root::Prefetch?j+??ݓ?!Z?	qV~O@)?j+??ݓ?1Z?	qV~O@:Preprocessing2E
Iterator::Root? ?	???!      Y@)?+e?X??1?????B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?n???@Q?\ywqX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d@?z?G??d@?z?G??!d@?z?G??      ??!       "	?? ?y`@?? ?y`@!?? ?y`@*      ??!       2      ??!       :	????1@????1@!????1@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?n???@y?\ywqX@?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2D#?&??9??!#?&??9??"E
'sequential/xception/block1_conv1/Conv2DConv2D??ʳ?!??!R??!????0"-
IteratorGetNext/_2_Recv?E??????!:??????"E
'sequential/xception/block1_conv2/Conv2DConv2D?;L{???!?_gr?@??0"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative?a??r??!??\????"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2D?+??X4??!?q?????"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2D.?ۤn???!E굹????"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2D?p?ϒ???!\ح????"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx?髬%??!̪???0??"i
>sequential/xception/block3_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNativeڸ0\??!Z?T??q??Q      Y@Y????S@a>4և??W@q>B?3ʉW@y?δ?&s[?"?

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