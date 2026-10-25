# SignBridge — نظام تجريبي للتعرّف على أوضاع اليد

SignBridge مشروع رؤية حاسوبية يستخدم كاميرا الويب وMediaPipe لاستخراج 21 نقطة من اليد، ثم يصنّفها باستخدام نموذج تعلم آلي. يتضمن تجربة بحثية لمقارنة الإحداثيات الخام بالإحداثيات النسبية المرتبطة بالرسغ.

> **تنبيه مهم:** SIGN_1 وSIGN_2 وSIGN_3 أسماء لأوضاع يد تجريبية وليست كلمات موثقة من لغة الإشارة المصرية. لا يُعد هذا الإصدار مترجمًا كاملًا للغة الإشارة.

## المميزات
- واجهة سطح مكتب باللغة العربية.
- تتبّع اليد في الوقت الحقيقي عبر كاميرا الويب.
- عرض الفئة المتوقعة ودرجة ثقة المصنف عند توفرها.
- نطق وصف تجريبي للنتيجة إذا كانت ميزة الصوت متاحة.
- جمع بيانات مع تسجيل فئة الإشارة وموضع اليد.
- تدريب نموذج الإحداثيات النسبية.
- تقييم RAW مقابل RELATIVE مع حجب موضع كاميرا كامل في كل مرة.

## متطلبات التشغيل
- Windows 10 أو Windows 11.
- Python 3.11 (موصى به).
- كاميرا ويب.
- اتصال إنترنت لتثبيت المكتبات أول مرة.

## تثبيت سريع
1. اضغط Code ثم Download ZIP في GitHub، وبعدها فك الضغط.
2. افتح المجلد بعد فك الضغط.
3. شغّل setup.bat وانتظر تثبيت المكتبات.
4. شغّل run_app.bat لفتح الواجهة.

يمكن أيضًا تشغيل الأوامر يدويًا من موجه الأوامر داخل المجلد:

    py -3.11 -m venv .venv
    .venv\\Scripts\\python.exe -m pip install --upgrade pip
    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
    .venv\\Scripts\\python.exe app.py

## الملفات الرئيسية
- app.py: الواجهة والتعرّف المباشر بالكاميرا.
- collect_research_data.py: جمع بيانات البحث للفئة والموضع المختارين.
- compare_models.py: مقارنة الإحداثيات الخام والنسبية.
- train_research_model.py: تدريب النموذج النسبي من research_data.csv.
- check_research_data.py: فحص سلامة البيانات.
- test_live_model.py: اختبار مباشر بالكاميرا من نافذة OpenCV.
- evaluate_external.py: تقييم النماذج على بيانات جديدة (شخص/إضاءة مختلفة).
- preflight.py / preflight.bat: فحص جاهزية اللابتوب قبل العرض.
- model_relative.pkl: النموذج النسبي المستخدم في التطبيق.
- model_raw.pkl: نموذج الإحداثيات الخام للمقارنة.
- research_data.csv: بيانات التجربة الحالية.

## تشغيل التطبيق
شغّل run_app.bat. لو لم يفتح النموذج، تأكد أن model_relative.pkl موجود بجوار app.py. عند ظهور خطأ كاميرا، أغلق أي برنامج آخر يستخدم الكاميرا وتأكد من صلاحيات الكاميرا في Windows.

## جمع بيانات جديدة
1. شغّل collect_research_data.bat أو الأمر: python collect_research_data.py.
2. اكتب اسم الفئة، مثل SIGN_1.
3. اختر موضع اليد من الخيارات الظاهرة.
4. حافظ على وضع اليد نفسه واضغط S لبدء الجمع. اضغط Q للخروج.
5. كرر الجمع لكل فئة في كل موضع. من الأفضل التسجيل في جلسات وأيام مختلفة.

الجامع يضيف العينات إلى research_data.csv في مجلد المشروع. احتفظ بنسخة احتياطية من بياناتك قبل جمع بيانات جديدة.

## إعادة تدريب النموذج
بعد التأكد من أن البيانات تحتوي فئتين على الأقل، شغّل:

    .venv\\Scripts\\python.exe train_research_model.py

سيُحفظ النموذج الناتج في model_relative.pkl، وهذا يستبدل النموذج النسبي الحالي؛ احتفظ بنسخة إذا أردت الرجوع إلى النموذج السابق.

## إعادة التجربة البحثية
شغّل check_research_data.py ثم compare_models.py. يقوم سكربت المقارنة بتدريب نموذج جديد مع حجب موضع كامل من التدريب واختباره على هذا الموضع. هذا يدرس حساسية النموذج لمكان اليد، لكنه لا يثبت التعميم على أشخاص جدد إذا كانت البيانات من شخص واحد.

## تفسير النتائج والقيود
- نسبة الثقة التي تعرضها الواجهة ليست ضمانًا أن التنبؤ صحيح.
- العينات المأخوذة من إطارات متتابعة قد تكون مترابطة؛ الدقة قد تبدو أعلى من الأداء الحقيقي.
- للحصول على دليل أقوى، اجمع جلسات مستقلة واختبر على أشخاص لم يشاركوا في التدريب.
- لا تربط أسماء الإشارات بكلمات فعلية إلا بعد التحقق من قاموس موثوق ومراجعة مستخدمين أو مختصين بلغة الإشارة المصرية.
- هذا نموذج تعليمي بحثي وليس أداة تواصل مضمونة أو بديلًا لمترجم لغة إشارة.

## النتائج

نُفّذت التجربة بـ `compare_models.py` (حجب موضع كاميرا كامل، 3 تجارب بقيم seed مختلفة، 1500 عينة في 15 جلسة). تُحفظ التفاصيل والرسوم في مجلد `results/`.

| المقياس | RAW | RELATIVE |
|---|---|---|
| متوسط الدقة | 61.6% | 99.98% |
| أسوأ موضع | 33.3% | 99.9% |
| زمن التنبؤ | ≈ 10 ms | ≈ 10 ms |

بيانات التجربة من شخص واحد، فهذه نتيجة أولية عن حساسية النموذج لمكان اليد وليست دليلًا على التعميم لأشخاص جدد. لاختبار بيانات جديدة: `python collect_research_data.py --out new_signer.csv` ثم `python evaluate_external.py new_signer.csv`. وقبل العرض شغّل `preflight.bat`، وراجع `DEMO_GUIDE_AR.md`.

## Egyptian Sign Language model mode (separate from the original pose demo)

A separate desktop mode is available in `esl_app.py`. It downloads the third-party model artifacts on first launch and uses a word-level Egyptian Sign Language research model with 56 labels. It does not replace the original `app.py` or the experimental SIGN_1/2/3 model.

### Windows quick start

1. Install Python 3.11 if it is not already installed.
2. Double-click `run_esl_app.bat`. It creates a separate `.venv-esl` environment and installs `requirements-esl.txt`.
3. Press **تشغيل النظام** in the app. The first launch downloads the model files and requires an internet connection.
4. Keep `THIRD_PARTY_NOTICES.md` with the project and review the limitations before presenting accuracy claims.

**Important:** this mode uses a third-party research model; it is not a complete sentence-level translator and its confidence score is not a verified accuracy percentage. Test the camera and model on this laptop before the presentation. The existing `run_app.bat` remains unchanged.
