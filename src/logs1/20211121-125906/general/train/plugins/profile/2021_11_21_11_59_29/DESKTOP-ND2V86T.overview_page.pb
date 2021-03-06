?	????ݙ?@????ݙ?@!????ݙ?@	?A????1@?A????1@!?A????1@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????ݙ?@E?a?H??1??)?5?|@IS#?3?j@YU?g$B?X@r0*	??C???@2O
Iterator::Root::PrefetchD3O?)iY@!??D??X@)D3O?)iY@1??D??X@:Preprocessing2E
Iterator::Root+?? jY@!      Y@)??\??ʎ?1??gJ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 17.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?A????1@I???????QIeLFIMT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E?a?H??E?a?H??!E?a?H??      ??!       "	??)?5?|@??)?5?|@!??)?5?|@*      ??!       2      ??!       :	S#?3?j@S#?3?j@!S#?3?j@B      ??!       J	U?g$B?X@U?g$B?X@!U?g$B?X@R      ??!       Z	U?g$B?X@U?g$B?X@!U?g$B?X@b      ??!       JGPUY?A????1@b q???????yIeLFIMT@?"J
,sequential_1/efficientnetb7/stem_conv/Conv2DConv2D??J+>??!??J+>??0"-
IteratorGetNext/_1_Send2?ݬ3??!?ؙ!??"R
9sequential_1/efficientnetb7/block2a_expand_activation/mulMulo??t????!*?.=??"_
4sequential_1/efficientnetb7/block3a_dwconv/depthwiseDepthwiseConv2dNative;ջo=?~?!щ????"K
2sequential_1/efficientnetb7/block2a_dwconv_pad/PadPad??f??}?!?<?$*٪?"_
4sequential_1/efficientnetb7/block2e_dwconv/depthwiseDepthwiseConv2dNative?@51?8|?!??KE`??"_
4sequential_1/efficientnetb7/block2c_dwconv/depthwiseDepthwiseConv2dNative?????{?!Oĳ?-???"Z
=sequential_1/efficientnetb7/block2a_expand_activation/SigmoidSigmoid?*O=F{?!?d?đ???"d
>sequential_1/efficientnetb7/block2a_expand_bn/FusedBatchNormV3FusedBatchNormV3|?h?z?!KL2??<??"_
4sequential_1/efficientnetb7/block2f_dwconv/depthwiseDepthwiseConv2dNative??????y?!??{??۵?Q      Y@Y?q??q?@asԸr?8X@q?=?uWP@y??cRюH?"?

both?Your program is MODERATELY input-bound because 17.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?65.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 