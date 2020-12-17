# أوراق TensorFlow البيضاء

يحدد هذا المستند المستندات التقنية حول TensorFlow.

## تعلم الآلة على نطاق واسع على أنظمة موزعة غير متجانسة

[الوصول إلى هذا المستند التعريفي التمهيدي.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**الملخص:** TensorFlow هي واجهة للتعبير عن خوارزميات التعلم الآلي وتنفيذ لتنفيذ مثل هذه الخوارزميات. يمكن تنفيذ الحساب المعبر عنه باستخدام TensorFlow مع تغيير طفيف أو بدون تغيير على مجموعة متنوعة من الأنظمة غير المتجانسة ، بدءًا من الأجهزة المحمولة مثل الهواتف والأجهزة اللوحية وحتى الأنظمة الموزعة على نطاق واسع لمئات الأجهزة وآلاف الأجهزة الحسابية مثل بطاقات GPU . النظام مرن ويمكن استخدامه للتعبير عن مجموعة متنوعة من الخوارزميات ، بما في ذلك خوارزميات التدريب والاستدلال لنماذج الشبكات العصبية العميقة ، وقد تم استخدامه لإجراء البحوث ونشر أنظمة التعلم الآلي في الإنتاج عبر أكثر من عشرة مجالات من علوم الكمبيوتر ومجالات أخرى ، بما في ذلك التعرف على الكلام ، ورؤية الكمبيوتر ، والروبوتات ، واسترجاع المعلومات ، ومعالجة اللغة الطبيعية ، واستخراج المعلومات الجغرافية ، واكتشاف العقاقير الحسابية. تصف هذه الورقة واجهة TensorFlow وتنفيذ تلك الواجهة التي أنشأناها في Google. تم إصدار TensorFlow API وتطبيق مرجعي كحزمة مفتوحة المصدر بموجب ترخيص Apache 2.0 في نوفمبر 2015 وهي متاحة على www.tensorflow.org.

### بتنسيق BibTeX

إذا كنت تستخدم TensorFlow في بحثك وترغب في الاستشهاد بنظام TensorFlow ، فنحن نقترح عليك الاستشهاد بهذا المستند التقني.

<pre>@misc{tensorflow2015-whitepaper,<br>title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},<br>url={https://www.tensorflow.org/},<br>note={Software available from tensorflow.org},<br>author={<br>    Mart\'{\i}n~Abadi and<br>    Ashish~Agarwal and<br>    Paul~Barham and<br>    Eugene~Brevdo and<br>    Zhifeng~Chen and<br>    Craig~Citro and<br>    Greg~S.~Corrado and<br>    Andy~Davis and<br>    Jeffrey~Dean and<br>    Matthieu~Devin and<br>    Sanjay~Ghemawat and<br>    Ian~Goodfellow and<br>    Andrew~Harp and<br>    Geoffrey~Irving and<br>    Michael~Isard and<br>    Yangqing Jia and<br>    Rafal~Jozefowicz and<br>    Lukasz~Kaiser and<br>    Manjunath~Kudlur and<br>    Josh~Levenberg and<br>    Dandelion~Man\'{e} and<br>    Rajat~Monga and<br>    Sherry~Moore and<br>    Derek~Murray and<br>    Chris~Olah and<br>    Mike~Schuster and<br>    Jonathon~Shlens and<br>    Benoit~Steiner and<br>    Ilya~Sutskever and<br>    Kunal~Talwar and<br>    Paul~Tucker and<br>    Vincent~Vanhoucke and<br>    Vijay~Vasudevan and<br>    Fernanda~Vi\'{e}gas and<br>    Oriol~Vinyals and<br>    Pete~Warden and<br>    Martin~Wattenberg and<br>    Martin~Wicke and<br>    Yuan~Yu and<br>    Xiaoqiang~Zheng},<br>  year={2015},<br>}</pre>

أو في شكل نصي:

<pre>Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,<br>Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,<br>Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,<br>Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,<br>Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,<br>Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,<br>Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,<br>Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,<br>Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,<br>Yuan Yu, and Xiaoqiang Zheng.<br>TensorFlow: Large-scale machine learning on heterogeneous systems,<br>2015. Software available from tensorflow.org.</pre>

## TensorFlow: نظام لتعلم الآلة على نطاق واسع

[الوصول إلى هذا المستند التعريفي التمهيدي.](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**الملخص:** TensorFlow هو نظام تعلم آلي يعمل على نطاق واسع وفي بيئات غير متجانسة. يستخدم TensorFlow الرسوم البيانية لتدفق البيانات لتمثيل الحساب والحالة المشتركة والعمليات التي تغير هذه الحالة. إنه يرسم عقد الرسم البياني لتدفق البيانات عبر العديد من الأجهزة في مجموعة ، وداخل جهاز عبر العديد من الأجهزة الحسابية ، بما في ذلك وحدات المعالجة المركزية متعددة النواة ، ووحدات معالجة الرسومات للأغراض العامة ، و ASICs المصممة خصيصًا والمعروفة باسم وحدات معالجة Tensor (TPUs). تمنح هذه البنية المرونة لمطور التطبيق: بينما في تصميمات "خادم المعلمات" السابقة ، تم دمج إدارة الحالة المشتركة في النظام ، يتيح TensorFlow للمطورين تجربة تحسينات جديدة وخوارزميات التدريب. يدعم TensorFlow مجموعة متنوعة من التطبيقات ، مع التركيز على التدريب والاستدلال على الشبكات العصبية العميقة. تستخدم العديد من خدمات Google TensorFlow في الإنتاج ، وقد أصدرناه كمشروع مفتوح المصدر ، وأصبح يستخدم على نطاق واسع في أبحاث التعلم الآلي. في هذا البحث ، نصف نموذج تدفق بيانات TensorFlow ونوضح الأداء المقنع الذي يحققه TensorFlow للعديد من تطبيقات العالم الحقيقي.
