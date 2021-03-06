?	,?`p?Ƒ@,?`p?Ƒ@!,?`p?Ƒ@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails',?`p?Ƒ@ۆQ<?@1??[˔??@I??l???@r0*	433333L@2O
Iterator::Root::Prefetch??ܥ?!??c?R@)??ܥ?1??c?R@:Preprocessing2E
Iterator::Root?!??u???!      Y@)???߾??1͋??pJ8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ?????Q?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ۆQ<?@ۆQ<?@!ۆQ<?@      ??!       "	??[˔??@??[˔??@!??[˔??@*      ??!       2      ??!       :	??l???@??l???@!??l???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?????y?????X@?"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative?'?{?(??!?'?{?(??"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2Dʵ} q??!??P|?̹?"i
>sequential/xception/block2_sepconv1/separable_conv2d/depthwiseDepthwiseConv2dNativeY??????!F?B-????"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2Df???????!???ɷ???"i
>sequential/xception/block3_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative?oA????!???????"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2DyxE1?ߓ?!?xh??(??"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx???8?&??!?y????"]
7sequential/xception/block2_sepconv2_bn/FusedBatchNormV3FusedBatchNormV3??????!/i?<???"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2D?) ?Z???!anߖ@??"P
4sequential/xception/block4_sepconv1/separable_conv2dConv2D?WQ?]??!?(?0????Q      Y@Y?RJ?)@a?
X+`?W@q?\;an,X@y*V@,YS=?"?

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
Refer to the TF2 Profiler FAQb?96.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 