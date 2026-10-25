# Third-party model and code notice

The optional Egyptian Sign Language recognition mode in `esl_app.py` downloads these model artifacts at first launch from the public repository:

- Project: ESL-software-ml — Real-Time Egyptian Sign Language Recognition
- Repository: https://github.com/abdullahsherdy/ESL-software-ml
- License stated by that repository: MIT
- Model artifacts: `model_v2.keras`, `label2idx.json`, and `holistic_landmarker.task`
- The upstream repository lists 56 labels in its label map. This is a word-level research model, not a full sentence translator.

The MIT license notice must be retained with redistributed copies. Review the upstream repository and its license before public/commercial redistribution.

Important evaluation note: the reported model score is the classifier's confidence output, not a guarantee that the predicted Egyptian Sign Language word is correct. Validate the application with independent signers and a suitable test set before claiming production accuracy.
