?	t~??@?q@t~??@?q@!t~??@?q@	?ɡ?????ɡ????!?ɡ????"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0t~??@?q@WAt?+@1+?gzIzq@I????;?@Y?o??e1??r0*	??????=@2E
Iterator::Root???_vO??!      Y@)X9??v???1?S??.J@:Preprocessing2O
Iterator::Root::Prefetch?!??u???!??L?G@)?!??u???1??L?G@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?ɡ????I??Fv?@Q:??=ZX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	WAt?+@WAt?+@!WAt?+@      ??!       "	+?gzIzq@+?gzIzq@!+?gzIzq@*      ??!       2      ??!       :	????;?@????;?@!????;?@B      ??!       J	?o??e1???o??e1??!?o??e1??R      ??!       Z	?o??e1???o??e1??!?o??e1??b      ??!       JGPUY?ɡ????b q??Fv?@y:??=ZX@?"F
(sequential/resnet152v2/conv1_conv/Conv2DConv2D???`????!???`????0"-
IteratorGetNext/_1_Sendz4O??g??!??G-w+??"O
1sequential/resnet152v2/conv2_block1_2_conv/Conv2DConv2D??f?T??!??zV??0"O
1sequential/resnet152v2/conv5_block1_0_conv/Conv2DConv2D?堣?Y??!?}.Ha??0"O
1sequential/resnet152v2/conv4_block1_0_conv/Conv2DConv2D_޾ul?{?!l????0"O
1sequential/resnet152v2/conv5_block1_2_conv/Conv2DConv2DI[#Ò?{?!!;"8ٷ?0"O
1sequential/resnet152v2/conv5_block3_2_conv/Conv2DConv2D?I^w:?{?!???{???0"O
1sequential/resnet152v2/conv5_block2_2_conv/Conv2DConv2D_K\??{?!??Eo?K??0"O
1sequential/resnet152v2/conv2_block2_3_conv/Conv2DConv2Dw|73:z?!xMy?G???0"P
2sequential/resnet152v2/conv4_block20_2_conv/Conv2DConv2D?????y?!????T???0Q      Y@Y]t?E??a?.?袋X@qI??)@y'U?wU?"?

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
Refer to the TF2 Profiler FAQb?13.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 