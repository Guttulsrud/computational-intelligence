?	?}V??q@?}V??q@!?}V??q@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?}V??q@?w(
????1??R%??p@I?????j	@r0*	P??n?=@2O
Iterator::Root::Prefetchw?Nyt??!??fS?O@)w?Nyt??1??fS?O@:Preprocessing2E
Iterator::Root-z??y??!      Y@)Sh?
??1b???B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?oV4߲??QA?.?4?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?w(
?????w(
????!?w(
????      ??!       "	??R%??p@??R%??p@!??R%??p@*      ??!       2      ??!       :	?????j	@?????j	@!?????j	@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?oV4߲??yA?.?4?X@?"F
(sequential/resnet152v2/conv1_conv/Conv2DConv2D?s?KS???!?s?KS???0"-
IteratorGetNext/_2_Recv?h?U???!2n?Pn???"O
1sequential/resnet152v2/conv5_block1_0_conv/Conv2DConv2D?R?gi?}?!??}?>??0"O
1sequential/resnet152v2/conv4_block1_0_conv/Conv2DConv2D?q?'%?}?!???" ???0"O
1sequential/resnet152v2/conv5_block3_2_conv/Conv2DConv2D???:p?|?!?WȰ?0"O
1sequential/resnet152v2/conv5_block2_2_conv/Conv2DConv2D??+??|?!C˪?$???0"O
1sequential/resnet152v2/conv5_block1_2_conv/Conv2DConv2Dʰ??"?|?!P??!?_??0"O
1sequential/resnet152v2/conv5_block3_1_conv/Conv2DConv2D2?m 	?y?!?`??????0"O
1sequential/resnet152v2/conv3_block1_0_conv/Conv2DConv2D
B?*?y?!?tV?????0"O
1sequential/resnet152v2/conv2_block2_2_conv/Conv2DConv2Dg?{3??x?!?1????0Q      Y@YH%?e??a?jch??X@qVѥڴW@yB?!??M?"?

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
Refer to the TF2 Profiler FAQb?94.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 