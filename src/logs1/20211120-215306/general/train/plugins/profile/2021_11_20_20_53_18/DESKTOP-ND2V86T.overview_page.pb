?	O!W?@O!W?@!O!W?@	?Ό??SF@?Ό??SF@!?Ό??SF@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0O!W?@????+???1??t?p@I? ??^.??Yin????k@r0*~j?t?UA)      ?=2O
Iterator::Root::Prefetchhur?b?k@!??=???X@)hur?b?k@1??=???X@:Preprocessing2E
Iterator::Root??ᰴ?k@!      Y@)
??ϛ???1?y ??Xr?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 44.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?Ό??SF@I?6?? ??Q2x??LK@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????+???????+???!????+???      ??!       "	??t?p@??t?p@!??t?p@*      ??!       2      ??!       :	? ??^.??? ??^.??!? ??^.??B      ??!       J	in????k@in????k@!in????k@R      ??!       Z	in????k@in????k@!in????k@b      ??!       JGPUY?Ό??SF@b q?6?? ??y2x??LK@?"H
*sequential_2/resnet152v2/conv1_conv/Conv2DConv2D"??(?Q??!"??(?Q??0"-
IteratorGetNext/_1_Senda?)?H͒?!R?1?h???"Q
3sequential_2/resnet152v2/conv5_block1_0_conv/Conv2DConv2Dq??ف?~?! 5a?ű?0"Q
3sequential_2/resnet152v2/conv4_block1_0_conv/Conv2DConv2Di??q~?!?E?qC???0"Q
3sequential_2/resnet152v2/conv5_block2_2_conv/Conv2DConv2D??c? }?!???{%w??0"Q
3sequential_2/resnet152v2/conv5_block1_2_conv/Conv2DConv2D?9????|?!^F??3F??0"Q
3sequential_2/resnet152v2/conv5_block3_2_conv/Conv2DConv2D	?T)??|?!o??J-??0"Q
3sequential_2/resnet152v2/conv3_block1_0_conv/Conv2DConv2D?7e??Lz?!??w?????0"Q
3sequential_2/resnet152v2/conv2_block2_2_conv/Conv2DConv2D?M?ڿx?!???t?D??0"Q
3sequential_2/resnet152v2/conv2_block1_2_conv/Conv2DConv2D?O5?קx?!?0?wϽ?0Q      Y@Y?r?? ]@a?h!?-*W@q$$	_X:@y(e???Od?"?

host?Your program is HIGHLY input-bound because 44.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?26.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 