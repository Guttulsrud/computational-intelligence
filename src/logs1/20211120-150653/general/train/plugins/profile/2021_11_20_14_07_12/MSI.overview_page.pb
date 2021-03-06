?	??Y?m?n@??Y?m?n@!??Y?m?n@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Y?m?n@?2????)@1^?$*l@A???K?'??I???}??@r0*	??v??zL@2E
Iterator::Root?}V?)??!      Y@)?x"??p??1͢7?O@:Preprocessing2O
Iterator::Root::Prefetch????Kq??!?2]??aB@)????Kq??1?2]??aB@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?8@?H? @Q??7???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2????)@?2????)@!?2????)@      ??!       "	^?$*l@^?$*l@!^?$*l@*      ??!       2	???K?'?????K?'??!???K?'??:	???}??@???}??@!???}??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?8@?H? @y??7???V@?"F
(sequential/resnet152v2/conv1_conv/Conv2DConv2D?DR?ޢ?!?DR?ޢ?0"-
IteratorGetNext/_2_Recv?])?i??!?(73g$??"O
1sequential/resnet152v2/conv4_block9_3_conv/Conv2DConv2Dg\FH????!^????y??0"O
1sequential/resnet152v2/conv5_block1_0_conv/Conv2DConv2D???a?}?!??n?=Y??0"O
1sequential/resnet152v2/conv3_block1_0_conv/Conv2DConv2D?9&??!}?!H)?$X+??0"O
1sequential/resnet152v2/conv4_block1_0_conv/Conv2DConv2DNhOI?|?!(???|???0"Q
1sequential/resnet152v2/conv5_block1_2_conv/Conv2DConv2D?^0d??|?!?*?ɼ?08"Q
1sequential/resnet152v2/conv5_block2_2_conv/Conv2DConv2Dix\?u?|?!?{`(????08"Q
1sequential/resnet152v2/conv5_block3_2_conv/Conv2DConv2Dk_Rt?|?![6ö+2??08"F
+sequential/resnet152v2/conv2_block2_out/addAddV23????x?!???????Q      Y@Y???????a??XؑX@qc{????W@y????M?"?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 