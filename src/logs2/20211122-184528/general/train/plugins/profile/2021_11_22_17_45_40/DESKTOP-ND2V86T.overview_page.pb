Ä	:#J{#a@:#J{#a@!:#J{#a@      øÿ!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails':#J{#a@¿sôæ?1WèelÊ`@IåÑ°¨ @r0*	      A@2O
Iterator::Root::Prefetchö_L?!N@)ö_L?1N@:Preprocessing2E
Iterator::RootÄ °rh¡?!      Y@)F%u?1jiiiiiC@:Preprocessing:«
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
ÅData preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
ÒReading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
ÅReading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
ºOther data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisÃ
deviceYour program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI Ãî>> @Qç¦~X@Zno#You may skip the rest of this page.B¼
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown÷
	¿sôæ?¿sôæ?!¿sôæ?      øÿ!       "	WèelÊ`@WèelÊ`@!WèelÊ`@*      øÿ!       2      øÿ!       :	åÑ°¨ @åÑ°¨ @!åÑ°¨ @B      øÿ!       J      øÿ!       R      øÿ!       Z      øÿ!       b      øÿ!       JGPUb q Ãî>> @yç¦~X@Ð"-
IteratorGetNext/_2_RecvVoy1u©?!Voy1u©?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2D¤óÿ{Ð¨?!ø¶#¹?"E
'sequential/xception/block1_conv1/Conv2DConv2DÊ4^bþ¡?!®EèãÁ?0"E
'sequential/xception/block1_conv2/Conv2DConv2D{`V¿n?!½ÓÕôþÄ?0"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative¢?UY6?!{ ÀeÈ?"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2DB,Ç?!_`0gË?"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2Dål?!ü{óYÎ?"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2D8Z`?!~Ñ"WÐ?"P
4sequential/xception/block4_sepconv1/separable_conv2dConv2DC	äõ@?!"ccòfáÑ?"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormExðo<õIØ?!!*·ëÓ?Q      Y@Y»ÉÝän@aGd#²ÙW@qàY9ÒW@yÇWòs=R[?"Ä

deviceYour program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*
<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2ö
=type.googleapis.com/tensorflow.profiler.GenericRecommendation´
nono*¥Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQbª95.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 