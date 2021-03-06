?	C????[@C????[@!C????[@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'C????[@aO;?5?@1?!Z@I'3?Vz-@r0*	?VA@2O
Iterator::Root::Prefetch??{???!ě¯w?O@)??{???1ě¯w?O@:Preprocessing2E
Iterator::Root??G?3???!      Y@)˞6????1;d=P?uB@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI 9WJ?@Qp?Z?RW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	aO;?5?@aO;?5?@!aO;?5?@      ??!       "	?!Z@?!Z@!?!Z@*      ??!       2      ??!       :	'3?Vz-@'3?Vz-@!'3?Vz-@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q 9WJ?@yp?Z?RW@?"-
IteratorGetNext/_2_Recv?????ԯ?!?????ԯ?"R
6sequential_1/xception/block4_sepconv2/separable_conv2dConv2D?+~?Ʀ?!??ۆ?M??"R
6sequential_1/xception/block2_sepconv2/separable_conv2dConv2D'RL0ȗ??!lwI?Y??"k
@sequential_1/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative{׍뿜?!
'??????"S
7sequential_1/xception/block14_sepconv2/separable_conv2dConv2Dqް&tr??!?B_/, ??"G
)sequential_1/xception/block1_conv2/Conv2DConv2D5]tɖ?!_?Z???0"R
6sequential_1/xception/block3_sepconv2/separable_conv2dConv2D?U?g4??!?D?????"k
@sequential_1/xception/block3_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNativeNwj'c???!??(??"S
7sequential_1/xception/block13_sepconv2/separable_conv2dConv2D+ϵ??d??!?k??_??"R
6sequential_1/xception/block2_sepconv1/separable_conv2dConv2D????j??!?{??u??Q      Y@YD?a?Y?@a??Id?W@q1?5??X@yv?f&?h?"?
both?Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?99.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 