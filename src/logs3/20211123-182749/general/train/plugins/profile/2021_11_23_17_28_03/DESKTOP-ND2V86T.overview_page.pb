?	?O ň7r@?O ň7r@!?O ň7r@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?O ň7r@????9??1?w??xr@I??r?m?@r0*	?z?G@@2O
Iterator::Root::Prefetch???r???!~??D'?N@)???r???1~??D'?N@:Preprocessing2E
Iterator::Root28J^?c??!      Y@)Q??lu??1?v??jC@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?3?؁%??Q2??i?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????9??????9??!????9??      ??!       "	?w??xr@?w??xr@!?w??xr@*      ??!       2      ??!       :	??r?m?@??r?m?@!??r?m?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?3?؁%??y2??i?X@?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2D	?ԑ\Ӫ?!	?ԑ\Ӫ?"-
IteratorGetNext/_2_Recv?rɵ???!???-	???"E
'sequential/xception/block1_conv2/Conv2DConv2D????f%??!HO??bD??0"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2D<}?O???!L??^?5??"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative弆??z??!?n?3????"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx~+??q/??!Y&mȪ??"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2D
cc??[??!??d>v??"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2D??pW??![h,A??"Q
5sequential/xception/block13_sepconv2/separable_conv2dConv2D??]q?<??!m??e??"Q
5sequential/xception/block11_sepconv1/separable_conv2dConv2D܄??j??!????;??Q      Y@Y?RJ?)@a?
X+`?W@qi`AX@yYԧ?`?"?

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
Refer to the TF2 Profiler FAQb?96.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 