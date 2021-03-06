?	?HZ?q@?HZ?q@!?HZ?q@	"???km??"???km??!"???km??"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?HZ?q@]?????1?~NA~?p@IwH1@????Y5]Ot]???r0*	?&1?|=@2O
Iterator::Root::Prefetch??eO???!?B?+?;P@)??eO???1?B?+?;P@:Preprocessing2E
Iterator::Roott???1??!      Y@)&?v??-??1?z????A@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9!???km??I l?2?X??QFk?<3?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?????]?????!]?????      ??!       "	?~NA~?p@?~NA~?p@!?~NA~?p@*      ??!       2      ??!       :	wH1@????wH1@????!wH1@????B      ??!       J	5]Ot]???5]Ot]???!5]Ot]???R      ??!       Z	5]Ot]???5]Ot]???!5]Ot]???b      ??!       JGPUY!???km??b q l?2?X??yFk?<3?X@?"P
4sequential/xception/block4_sepconv2/separable_conv2dConv2DPR??t??!PR??t??"E
'sequential/xception/block1_conv1/Conv2DConv2D72?`???!DBt]????0"-
IteratorGetNext/_1_Send??R@????!???????"i
>sequential/xception/block2_sepconv2/separable_conv2d/depthwiseDepthwiseConv2dNative????Dn??!z??;??"E
'sequential/xception/block1_conv2/Conv2DConv2D????@n??!yy<ވ??0"P
4sequential/xception/block3_sepconv2/separable_conv2dConv2Dt?{?????!??EY6???"P
4sequential/xception/block2_sepconv2/separable_conv2dConv2D'U??,ՙ?!-bX????"Q
5sequential/xception/block14_sepconv2/separable_conv2dConv2DT?VG:??!???.???"^
7sequential/xception/block2_sepconv1_bn/FusedBatchNormV3_FusedBatchNormEx?????;??!?VHx??"P
4sequential/xception/block4_sepconv1/separable_conv2dConv2D??HZΓ?!?k?-???Q      Y@Y?????n@aGd#??W@q?=}*?!@y????OH?"?	
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
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 