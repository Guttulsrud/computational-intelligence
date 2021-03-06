?	Mu@??@Mu@??@!Mu@??@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Mu@??@s?FZ*
P@1kb???}@A?&?|???I??p??C@r0*	n???J@2O
Iterator::Root::Prefetch?:TS?u??!!?Q??N@)?:TS?u??1!?Q??N@:Preprocessing2E
Iterator::RootH?9????!      Y@)i?X??1?? ?C@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIL???n2@Q???K?}T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s?FZ*
P@s?FZ*
P@!s?FZ*
P@      ??!       "	kb???}@kb???}@!kb???}@*      ??!       2	?&?|????&?|???!?&?|???:	??p??C@??p??C@!??p??C@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qL???n2@y???K?}T@?"-
IteratorGetNext/_2_Recv???ꁡ?!???ꁡ?"P
7sequential/efficientnetb7/block2a_expand_activation/mulMul??>*?8??!y;\???"R
4sequential/efficientnetb7/block7d_expand_conv/Conv2DConv2D?&A?3Q??!&?+Ef$??0"I
0sequential/efficientnetb7/block2a_dwconv_pad/PadPad???????!?x??jE??"b
<sequential/efficientnetb7/block2a_expand_bn/FusedBatchNormV3FusedBatchNormV3???p?^??!F??Gq??"]
2sequential/efficientnetb7/block3g_dwconv/depthwiseDepthwiseConv2dNative}H???Հ?!V?hX????"P
7sequential/efficientnetb7/block2g_expand_activation/mulMul?Lg1o???!???>Ȣ??"P
7sequential/efficientnetb7/block2d_expand_activation/mulMul|??}U?!?? ???"I
0sequential/efficientnetb7/block2e_activation/mulMul? ?VU~?! ??pu}??"Q
4sequential/efficientnetb7/block2f_activation/SigmoidSigmoid.9??}?!cc?rcV??Q      Y@Y|a????a?=????X@q???[?V@yU?Z??<?"?
both?Your program is POTENTIALLY input-bound because 11.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?91.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 