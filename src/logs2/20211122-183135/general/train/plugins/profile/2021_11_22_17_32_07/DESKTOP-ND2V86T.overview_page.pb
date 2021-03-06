?	<?(A?Os@<?(A?Os@!<?(A?Os@	n?????n?????!n?????"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9<?(A?Os@?:u???'@1?҆?RNr@A??Za?~?I&???L?@YE???J???r0*	???K7	I@2O
Iterator::Root::PrefetchE7????!T??"Q@)E7????1T??"Q@:Preprocessing2E
Iterator::Root????	???!      Y@)T㥛? ??1?rďgt?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9n?????I?ǝv?@Q?w%?\?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:u???'@?:u???'@!?:u???'@      ??!       "	?҆?RNr@?҆?RNr@!?҆?RNr@*      ??!       2	??Za?~???Za?~?!??Za?~?:	&???L?@&???L?@!&???L?@B      ??!       J	E???J???E???J???!E???J???R      ??!       Z	E???J???E???J???!E???J???b      ??!       JGPUYn?????b q?ǝv?@y?w%?\?W@?"H
*sequential_4/resnet152v2/conv1_conv/Conv2DConv2D?<q\??!?<q\??0"-
IteratorGetNext/_1_Send?fŠy??!+iϠL???"Q
3sequential_4/resnet152v2/conv5_block3_2_conv/Conv2DConv2Ds?Z??}?!2*|ff??0"Q
4sequential_4/resnet152v2/conv2_block1_3_conv/BiasAddBiasAdd??k0??{?!????t$??"Q
3sequential_4/resnet152v2/conv5_block1_0_conv/Conv2DConv2DP???z{?!??D3 ܶ?0"Q
3sequential_4/resnet152v2/conv4_block1_0_conv/Conv2DConv2D??F?s{?!?3?m????0"Q
3sequential_4/resnet152v2/conv5_block2_2_conv/Conv2DConv2D?$Y/eTz?!?Ů??1??0"Q
3sequential_4/resnet152v2/conv5_block1_2_conv/Conv2DConv2D?i"E?Qz?!*? ??ֻ?0"R
4sequential_4/resnet152v2/conv4_block34_2_conv/Conv2DConv2D??ݙS^y?!dŞ??l??0"R
4sequential_4/resnet152v2/conv4_block22_2_conv/Conv2DConv2D???R?9y?!l??| ??0Q      Y@Y?g?mɺ??aa?I??X@q??A?Q@y?~????U?"?

both?Your program is POTENTIALLY input-bound because 3.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?68.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 