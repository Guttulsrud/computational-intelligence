?	.2%*r@.2%*r@!.2%*r@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'.2%*r@?1^???1?6??A?q@I?s??{@r0*	\???(,B@2O
Iterator::Root::PrefetchW{?l??!?{ΥlwO@)W{?l??1?{ΥlwO@:Preprocessing2E
Iterator::Root?	??ϛ??!      Y@)?/K;5???1>?1Z??B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?s
?Y?@Qd??3erX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1^????1^???!?1^???      ??!       "	?6??A?q@?6??A?q@!?6??A?q@*      ??!       2      ??!       :	?s??{@?s??{@!?s??{@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?s
?Y?@yd??3erX@?"-
IteratorGetNext/_2_RecvM_?ع	??!M_?ع	??"Q
3sequential_1/resnet152v2/conv5_block3_2_conv/Conv2DConv2D?}?`a???! ?3??ɜ?0"Q
3sequential_1/resnet152v2/conv4_block8_2_conv/Conv2DConv2D?*|??;??!?ٸ??s??0"Q
3sequential_1/resnet152v2/conv3_block5_2_conv/Conv2DConv2DULK``???!׬˘?a??0"R
4sequential_1/resnet152v2/conv4_block12_2_conv/Conv2DConv2D??ƥ ???!Be=?uH??0"Q
3sequential_1/resnet152v2/conv4_block4_3_conv/Conv2DConv2D?K
????!??0????0"R
4sequential_1/resnet152v2/conv4_block28_1_conv/Conv2DConv2D&ϒ%???!V???<??0"Q
3sequential_1/resnet152v2/conv5_block2_2_conv/Conv2DConv2D???V?!????T??0"Q
3sequential_1/resnet152v2/conv5_block1_2_conv/Conv2DConv2D?X?2/???!
qmw?l??0"R
4sequential_1/resnet152v2/conv4_block16_1_conv/Conv2DConv2D??\?u??!?;{??0Q      Y@YH%?e??a?jch??X@q????X@y0Zd??P?"?

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