?	j?0
??b@j?0
??b@!j?0
??b@	?U?	?K???U?	?K??!?U?	?K??"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0j?0
??b@P?Lۿ2??1?%?Yb@In2?????Y?ٲ|]??r0*	??????B@2O
Iterator::Root::Prefetch???????!/????.O@)???????1/????.O@:Preprocessing2E
Iterator::RootΈ?????!      Y@)y?&1???1?DM4?B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?U?	?K??I???????QW???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	P?Lۿ2??P?Lۿ2??!P?Lۿ2??      ??!       "	?%?Yb@?%?Yb@!?%?Yb@*      ??!       2      ??!       :	n2?????n2?????!n2?????B      ??!       J	?ٲ|]???ٲ|]??!?ٲ|]??R      ??!       Z	?ٲ|]???ٲ|]??!?ٲ|]??b      ??!       JGPUY?U?	?K??b q???????yW???X@?"F
$sequential_1/vgg16/block1_conv2/Relu_FusedConv2D@?E,4W??!@?E,4W??"F
$sequential_1/vgg16/block2_conv2/Relu_FusedConv2DѦ?4|a??!TiI#????"F
$sequential_1/vgg16/block3_conv2/Relu_FusedConv2D?? ??!?.?????"F
$sequential_1/vgg16/block3_conv3/Relu_FusedConv2DkM???ܶ?!?.?>C??"F
$sequential_1/vgg16/block4_conv2/Relu_FusedConv2DjE??????!?)(?O??"F
$sequential_1/vgg16/block4_conv3/Relu_FusedConv2D`??C%???!??T???"F
$sequential_1/vgg16/block1_conv1/Relu_FusedConv2D?U?G?G??!?q??G???"F
$sequential_1/vgg16/block2_conv1/Relu_FusedConv2D?x?s$??!XtH????"F
$sequential_1/vgg16/block3_conv1/Relu_FusedConv2D?d0????!?z?P????"F
$sequential_1/vgg16/block4_conv1/Relu_FusedConv2D?bt????!??2??"??Q      Y@Y?X`?%@aP????OV@q????gR@yHη?`?"?

device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?73.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 