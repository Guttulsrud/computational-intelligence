?	;%??|@;%??|@!;%??|@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails';%??|@Ḍ?? @1???d?|@I?????@r0*	     ?@@2O
Iterator::Root::Prefetch/?$???!T?n?WO@)/?$???1T?n?WO@:Preprocessing2E
Iterator::Root???x?&??!      Y@)????????1??1??B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?ݻ?g??Q?y?c?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ḍ?? @Ḍ?? @!Ḍ?? @      ??!       "	???d?|@???d?|@!???d?|@*      ??!       2      ??!       :	?????@?????@!?????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ݻ?g??y?y?c?X@?"H
*sequential/efficientnetb7/stem_conv/Conv2DConv2Dti	_#D??!ti	_#D??0"P
7sequential/efficientnetb7/block2a_expand_activation/mulMulޜ??2??!??V?<˜?"-
IteratorGetNext/_2_Recv5+Mj??!???1 ??"I
0sequential/efficientnetb7/block2a_dwconv_pad/PadPad?? ???!aV?1???"]
2sequential/efficientnetb7/block3a_dwconv/depthwiseDepthwiseConv2dNative??>30|?!?)# ??"X
;sequential/efficientnetb7/block2a_expand_activation/SigmoidSigmoid??`N5?{?!?A??}???"b
<sequential/efficientnetb7/block2a_expand_bn/FusedBatchNormV3FusedBatchNormV38?U?l?z?!N?ܰ%???"]
2sequential/efficientnetb7/block2g_dwconv/depthwiseDepthwiseConv2dNative??Ƕ??y?!xI?????"]
2sequential/efficientnetb7/block2e_dwconv/depthwiseDepthwiseConv2dNative{?7Dy?!?/?$???"]
2sequential/efficientnetb7/block2d_dwconv/depthwiseDepthwiseConv2dNative?7b?:y?!s??:????Q      Y@Y??;?ϭ??a???`??X@qۇ?R??W@yע??xI?"?

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
Refer to the TF2 Profiler FAQb?94.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 