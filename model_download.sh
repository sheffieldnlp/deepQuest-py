#! /bin/sh


#urls for trained model downloads
MODELS_URL="https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_et_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_ro_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_si_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_ne_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_en_zh.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki25k_et_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_ro_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_si_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_ne_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_en_zh.tar.gz"


trainedmodelsdir=trained_models/

mkdir ${trainedmodelsdir}

for u in $MODELS_URL
do
   wget "${u}" -P ${trainedmodelsdir}
done
