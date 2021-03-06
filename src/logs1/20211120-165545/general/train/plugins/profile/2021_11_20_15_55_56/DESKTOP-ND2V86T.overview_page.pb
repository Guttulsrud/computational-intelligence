?	1?Tm??p@1?Tm??p@!1?Tm??p@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'1?Tm??p@?>????1?ʢ???p@I?U???V@r0*	>
ףp?L@2O
Iterator::Root::Prefetch??O?m??!P???xT@)??O?m??1P???xT@:Preprocessing2E
Iterator::Root!Y?n??!      Y@)5&?\R??1?ҕc?2@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI j?F?@Q???}AoX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?>?????>????!?>????      ??!       "	?ʢ???p@?ʢ???p@!?ʢ???p@*      ??!       2      ??!       :	?U???V@?U???V@!?U???V@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q j?F?@y???}AoX@?"H
*sequential_1/resnet152v2/conv1_conv/Conv2DConv2D׉V?\??!׉V?\??0"-
IteratorGetNext/_2_Recv??(o#???!??b]y??"Q
3sequential_1/resnet152v2/conv5_block1_0_conv/Conv2DConv2DvĬ9ƺ?!D??)?p??0"Q
3sequential_1/resnet152v2/conv4_block1_0_conv/Conv2DConv2D??~ ?!????P??0"Q
3sequential_1/resnet152v2/conv5_block3_2_conv/Conv2DConv2D=??_??}?!?W?B???0"Q
3sequential_1/resnet152v2/conv5_block2_2_conv/Conv2DConv2D?.???}?!?E?M?e??0"Q
3sequential_1/resnet152v2/conv5_block1_2_conv/Conv2DConv2D]?8??}?!f?C??0"Q
3sequential_1/resnet152v2/conv3_block1_0_conv/Conv2DConv2D?dۏ?y?!???F???0"Q
3sequential_1/resnet152v2/conv2_block1_2_conv/Conv2DConv2D,*???x?!Oe:Tn??0"Q
3sequential_1/resnet152v2/conv2_block2_2_conv/Conv2DConv2DL???l?x?!$u2?????0Q      Y@YH4:????a/g???X@q2?ص??X@y????W:H?"?

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
Refer to the TF2 Profiler FAQb?99.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 