?	C8fٓ.`@C8fٓ.`@!C8fٓ.`@	#xK?g???#xK?g???!#xK?g???"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C8fٓ.`@??¼Ǚ??1?R\U??_@I
?_禍??Y
?Y2???r0*	???Mb?=@2O
Iterator::Root::Prefetchh?.?K??!2Ճ??N@)h?.?K??12Ճ??N@:Preprocessing2E
Iterator::Root?*4?f??!      Y@)Q??Û5??1?*|?u?C@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9#xK?g???I??>g]??Q
?|"CsX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??¼Ǚ????¼Ǚ??!??¼Ǚ??      ??!       "	?R\U??_@?R\U??_@!?R\U??_@*      ??!       2      ??!       :	
?_禍??
?_禍??!
?_禍??B      ??!       J	
?Y2???
?Y2???!
?Y2???R      ??!       Z	
?Y2???
?Y2???!
?Y2???b      ??!       JGPUY#xK?g???b q??>g]??y
?|"CsX@?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2Dn???댩?!n???댩?"-
IteratorGetNext/_1_Send?Ҽ?{u??!??,?3???"E
'sequential/xception/block1_conv2/Conv2DConv2D%??y????!FMx}{??0"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative????,???!6@cVN??"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2D?m0?w???!?M	H?l??"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2D?Ҧ????!ϐ???k??"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2D?W? ]*??!?[??Q??"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx??ɺ?b??!Q?k???"Q
5sequential/xception/block13_sepconv2/separable_conv2dConv2D???0???!??6=????"P
4sequential/xception/block4_sepconv1/separable_conv2dConv2D?qv???!,?m?????Q      Y@Y?RJ?)@a?
X+`?W@qb?緒?@yG???e?"?	
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 