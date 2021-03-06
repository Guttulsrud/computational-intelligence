?	|???t?@|???t?@!|???t?@	Y?7?ƝN@Y?7?ƝN@!Y?7?ƝN@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0|???t?@?'v?@1?EC?#?r@In/??Yk??=]?}@r0*	??v>?fA2O
Iterator::Root::Prefetch??{s~@!?8M???X@)??{s~@1?8M???X@:Preprocessing2E
Iterator::Root<????~@!      Y@)???8???1?z?cY4a?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 61.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9Y?7?ƝN@I@??????Q?UɱPC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'v?@?'v?@!?'v?@      ??!       "	?EC?#?r@?EC?#?r@!?EC?#?r@*      ??!       2      ??!       :	n/??n/??!n/??B      ??!       J	k??=]?}@k??=]?}@!k??=]?}@R      ??!       Z	k??=]?}@k??=]?}@!k??=]?}@b      ??!       JGPUYY?7?ƝN@b q@??????y?UɱPC@?"H
*sequential_1/resnet152v2/conv1_conv/Conv2DConv2D>]4e?H??!>]4e?H??0"-
IteratorGetNext/_1_Send????忘?!?p?w7??"Q
3sequential_1/resnet152v2/conv4_block1_0_conv/Conv2DConv2D݆'?????!?R|5??0"R
4sequential_1/resnet152v2/conv4_block35_2_conv/Conv2DConv2De?>????!?F????0"Q
3sequential_1/resnet152v2/conv4_block8_3_conv/Conv2DConv2Dw?ȭ???!?%????0"Q
3sequential_1/resnet152v2/conv5_block1_0_conv/Conv2DConv2D??????!CGo???0"R
4sequential_1/resnet152v2/conv4_block27_3_conv/Conv2DConv2D	x\?*??!DI?љ>??0"R
4sequential_1/resnet152v2/conv4_block22_1_conv/Conv2DConv2D]???6??!*?mE??0"Q
3sequential_1/resnet152v2/conv4_block6_3_conv/Conv2DConv2DO??}?!e?i_??0"Q
3sequential_1/resnet152v2/conv2_block2_2_conv/Conv2DConv2Dm??G??|?!?3????0Q      Y@Y'M?4i?@a-[?l?W@q.x]?7?1@yP? (??e?"?

host?Your program is HIGHLY input-bound because 61.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?17.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 